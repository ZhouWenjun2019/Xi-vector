import torch
import torch.nn.functional as F
from utils.common_functions import to_device


class TdnnAffine(torch.nn.Module):
    """ An implemented tdnn affine component by conv1d
        y = splice(w * x, context) + b

    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g.  [-2,0,2]
    If context is [0], then the TdnnAffine is equal to linear layer.
    """
    def __init__(self, input_dim, output_dim, context=[0], bias=True, 
                 pad=True, stride=1, groups=1, norm_w=False, norm_f=False):
        super(TdnnAffine, self).__init__()
        assert input_dim % groups == 0
        # Check to make sure the context sorted and has no duplicated values
        for index in range(0, len(context) - 1):
            if(context[index] >= context[index + 1]):
                raise ValueError("Context tuple {} is invalid, such as the order.".format(context))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.groups = groups

        self.norm_w = norm_w
        self.norm_f = norm_f

        # It is used to subsample frames with this factor
        self.stride = stride

        self.left_context = context[0] if context[0] < 0 else 0 
        self.right_context = context[-1] if context[-1] > 0 else 0 

        self.tot_context = self.right_context - self.left_context + 1

        # Do not support sphereConv now.
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            print("Warning: do not support sphereConv now and set norm_f=False.")

        kernel_size = (self.tot_context,)

        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim//groups, *kernel_size))

        if self.bool_bias:
            self.bias = torch.nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)

        # init weight and bias. It is important
        self.init_weight()

        # Save GPU memory for no skiping case
        if len(context) != self.tot_context:
            # Used to skip some frames index according to context
            self.mask = torch.tensor([[[ 1 if index in context else 0 \
                                        for index in range(self.left_context, self.right_context + 1) ]]])
        else:
            self.mask = None

        ## Deprecated: the broadcast method could be used to save GPU memory, 
        # self.mask = torch.randn(output_dim, input_dim, 0)
        # for index in range(self.left_context, self.right_context + 1):
        #     if index in context:
        #         fixed_value = torch.ones(output_dim, input_dim, 1)
        #     else:
        #         fixed_value = torch.zeros(output_dim, input_dim, 1)

        #     self.mask=torch.cat((self.mask, fixed_value), dim = 2)

        # Save GPU memory of thi case.

        self.selected_device = False

    def init_weight(self):
        # Note, var should be small to avoid slow-shrinking
        torch.nn.init.normal_(self.weight, 0., 0.01)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Do not use conv1d.padding for self.left_context + self.right_context != 0 case.
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context), mode="constant", value=0)

        assert inputs.shape[2] >=  self.tot_context

        if not self.selected_device and self.mask is not None:
            # To save the CPU -> GPU moving time
            # Another simple case, for a temporary tensor, jus specify the device when creating it.
            # such as, this_tensor = torch.tensor([1.0], device=inputs.device)
            self.mask = to_device(self, self.mask)
            self.selected_device = True

        filters = self.weight  * self.mask if self.mask is not None else self.weight

        if self.norm_w:
            filters = F.normalize(filters, dim=1)

        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)

        outputs = F.conv1d(inputs, filters, self.bias, stride=self.stride, padding=0, dilation=1, groups=self.groups)

        return outputs

    def extra_repr(self):
        return '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, ' \
               'pad={pad}, groups={groups}, norm_w={norm_w}, norm_f={norm_f}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        x = x[0]

        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])

