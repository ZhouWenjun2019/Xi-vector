"""
Date Created: Feb 10 2020

This file contains the model descriptions, including original x-vector architecture. The first two models are in active developement. All others are provided below
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class xvecTDNN(nn.Module):

    def __init__(self, numSpkrs, featDim, p_dropout):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=featDim, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000, 512) # 添加hook，得到embedding
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512, numSpkrs)

    def forward(self, x, eps):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        # frame-level
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        # if self.training:
        #     shape = x.size()
        #     noise = torch.cuda.FloatTensor(shape)
        #     torch.randn(shape, out=noise)
        #     x += noise*eps

        # average pooling
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1) # 拼接

        # utt-level
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        # x = self.fc3(x) # Check utils.nnet.margins
        return x


class xivecTDNN(nn.Module):

    def __init__(self, numSpkrs, featDim, p_dropout):
        super(xivecTDNN, self).__init__()

        self.z0 = nn.Parameter(torch.zeros(1, 1500, 1), 
                               requires_grad=True)
        self.l0 = nn.Parameter(torch.zeros(1, 1500, 1), 
                               requires_grad=True)

        self.tdnn1 = nn.Conv1d(in_channels=featDim, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        # # fc in encoder
        # self.enc_fc = nn.Conv1d(in_channels=1500, out_channels=1500, kernel_size=1)
        # self.bn_enc_fc = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        # self.dropout_enc_fc = nn.Dropout(p=p_dropout)

        self.aux_fc1 = nn.Conv1d(in_channels=1500, out_channels=256, kernel_size=1)
        self.bn_aux_fc1 = nn.BatchNorm1d(256, momentum=0.1, affine=False)
        self.dropout_aux_fc1 = nn.Dropout(p=p_dropout)

        self.aux_fc2 = nn.Conv1d(in_channels=256, out_channels=1500, kernel_size=1)
        self.bn_aux_fc2 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_aux_fc2 = nn.Dropout(p=p_dropout)

        # fc in decoder
        self.fc1 = nn.Linear(1500, 512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        # fc in decoder
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512, numSpkrs)

    def forward(self, x, eps):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        # frame-level
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        
        # x = self.dropout_enc_fc(self.bn_enc_fc(F.relu(self.enc_fc(x))))

        # if self.training:
        #     shape = x.size()
        #     noise = torch.cuda.FloatTensor(shape)
        #     torch.randn(shape, out=noise)
        #     x += noise*eps

        # precision matrix
        lt = self.dropout_aux_fc1(self.bn_aux_fc1(F.relu(self.aux_fc1(x))))
        lt = self.dropout_aux_fc2(self.bn_aux_fc2(F.softplus(self.aux_fc2(lt))))
        lt = torch.cat((self.l0.expand(lt.size(0), -1, -1), lt), dim=2)
        x = torch.cat((self.z0.expand(lt.size(0), -1, -1), x), dim=2)
        # Gaussian Posterior Inference
        A = F.softmax(lt, dim=1) # 32*1500*(1+123)
        Phi = A.mul(x).sum(dim=2) # 32*1500

        # utt-level
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(Phi))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        # x = self.fc3(x) # Check utils.nnet.margins
        return x

