from __future__ import print_function
from __future__ import division
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np

class InnerProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575):
        super(InnerProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature # num_class
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight) 

    def forward(self, input, label):
        # label not used
        output = F.linear(input, self.weight) # 没有偏置
        return output


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: \n
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin 
        cos(theta + m)
    Refer to paper:
        CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        p_target = [0.9, 0.95, 0.99]
        suggested_s = [ (out_features-1)/out_features*np.log((out_features-1)*x/(1-x)) for x in p_target ]
        if self.s < suggested_s[0]:
            print("Warning : using feature noamlization with small scalar s={s} could result in bad convergence. There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
            s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label, aug_hard_samples=False):
        # cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # convert label to one-hot
        one_hot = torch.zeros_like(cosine, device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2))) 

        if aug_hard_samples:
            # 得到每个样本对应标签的cos值
            cos_label = torch.sum(one_hot.mul(cosine), dim=1)
            cos_increase_id = cos_label.argsort().tolist()

            # 升序排列，越靠后margin越小
            ###################################################################
            map1 = torch.zeros_like(cosine, device='cpu')
            map1.scatter_(1, label.view(-1, 1).long(), 1)
            left_limit1 = int(0 * len(cos_increase_id))
            right_limit1 = int(0.15 * len(cos_increase_id))
            selected_sample_id = cos_increase_id[left_limit1 : right_limit1]
            non_selected_sample_id = list(set(cos_increase_id)-set(selected_sample_id))
            map1[non_selected_sample_id, :] = 0

            m = 0.5
            cos_m = math.cos(m)
            sin_m = math.sin(m)
            th = math.cos(math.pi - m) # threshold
            mm = math.sin(math.pi - m) * m
            phi1 = cosine * cos_m - sine * sin_m
            if self.easy_margin: 
                phi1 = torch.where(cosine > 0, phi1, cosine)
            else:
                phi1 = torch.where(cosine > th, phi1, cosine-mm)

            ###################################################################
            map2 = torch.zeros_like(cosine, device='cpu')
            map2.scatter_(1, label.view(-1, 1).long(), 1)
            left_limit2 = int(0.15 * len(cos_increase_id))
            right_limit2 = int(0.85 * len(cos_increase_id))
            selected_sample_id = cos_increase_id[left_limit2 : right_limit2]
            non_selected_sample_id = list(set(cos_increase_id)-set(selected_sample_id))
            map2[non_selected_sample_id, :] = 0

            m = 0.25
            cos_m = math.cos(m)
            sin_m = math.sin(m)
            th = math.cos(math.pi - m) # threshold
            mm = math.sin(math.pi - m) * m
            phi2 = cosine * cos_m - sine * sin_m
            if self.easy_margin: 
                phi2 = torch.where(cosine > 0, phi2, cosine)
            else:
                phi2 = torch.where(cosine > th, phi2, cosine-mm)

            ###################################################################
            map3 = torch.zeros_like(cosine, device='cpu')
            map3.scatter_(1, label.view(-1, 1).long(), 1)
            left_limit3 = int(0.85 * len(cos_increase_id))
            right_limit3 = int(1.01 * len(cos_increase_id))
            selected_sample_id = cos_increase_id[left_limit3 : right_limit3]
            non_selected_sample_id = list(set(cos_increase_id)-set(selected_sample_id))
            map3[non_selected_sample_id, :] = 0

            m = 0
            cos_m = math.cos(m)
            sin_m = math.sin(m)
            th = math.cos(math.pi - m) # threshold
            mm = math.sin(math.pi - m) * m
            phi3 = cosine * cos_m - sine * sin_m
            if self.easy_margin: 
                phi3 = torch.where(cosine > 0, phi3, cosine)
            else:
                phi3 = torch.where(cosine > th, phi3, cosine-mm)

            ###################################################################
            # map4 = torch.zeros_like(cosine, device='cpu')
            # map4.scatter_(1, label.view(-1, 1).long(), 1)
            # left_limit4 = int(0.75 * len(cos_increase_id))
            # right_limit4 = int(1.01 * len(cos_increase_id))
            # selected_sample_id = cos_increase_id[left_limit4 : right_limit4]
            # non_selected_sample_id = list(set(cos_increase_id)-set(selected_sample_id))
            # map4[non_selected_sample_id, :] = 0

            # m = 0
            # cos_m = math.cos(m)
            # sin_m = math.sin(m)
            # th = math.cos(math.pi - m) # threshold
            # mm = math.sin(math.pi - m) * m
            # phi4 = cosine * cos_m - sine * sin_m
            # if self.easy_margin: 
            #     phi4 = torch.where(cosine > 0, phi4, cosine)
            # else:
            #     phi4 = torch.where(cosine > th, phi4, cosine-mm)

            output = map1 * phi1 + \
                     map2 * phi2 + \
                     map3 * phi3 + \
                     (1.0 - one_hot) * cosine

        else:
            phi = cosine * self.cos_m - sine * self.sin_m
            
            if self.easy_margin: # 不考虑到pi+m不满足单调递减的情况，其实也不会发生
                # cosine>0 means two class is similar
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine-self.mm)
            
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        if aug_hard_samples:
            return output, cos_label, cos_increase_id
        else:
            return output


class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: \n
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    Refer to paper:
        CosFace: Large Margin Cosine Loss for Deep Face Recognition.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # torch.where(out_i = {x_i if condition_i else y_i)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    Refer to paper:
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # cos(theta) & phi(theta) 
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # convert label to one-hot 
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # Calculate output
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'


class LargeMarginGMM(nn.Module):
    """ 
    Refer to paper:
        Rethinking Feature Distribution for Loss Functions in Image Classification
    Refer to code:
        https://github.com/YirongMao/softmax_variants
    """
    def __init__(self, in_features, out_features, alpha):
        super(LargeMarginGMM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.alpha = alpha

    def forward(self, input, label):


        pass


