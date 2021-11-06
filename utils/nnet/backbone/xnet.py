from utils.nnet.backbone.TDNN import StatsPooling, FullyConnected, TDNN
import torch.nn as nn

class xnet(nn.Module):
    def __init__(self):
        super(xnet, self).__init__()
        self.classifier = nn.Sequential(
            TDNN([-2, 2], 24, 512, full_context=True),
            TDNN([-2, 1, 2], 512, 512, full_context=False),
            TDNN([-3, 1, 3], 512, 512, full_context=False),
            TDNN([1], 512, 512, full_context=False),
            TDNN([1], 512, 1500, full_context=False),
            StatsPooling(),
            FullyConnected(),
            # Final = nn.Linear(512, 4).double() # 算在margin里面
        )

    def forward(self, x):
        '''
        提取embedding
        '''
        return self.classifier(x)
