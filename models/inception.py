from __future__ import absolute_import, division, print_function

import os
from os.path import join as pjoin

import torch
import torch.nn as nn

from .base import _Base, ConvTranspose3d, Conv3d, Merge, Split


class StemBlock(nn.Module):
    def __init__(self, return_indices=False, batch_norm=True):
        super(StemBlock,self).__init__()
        self.return_indices = return_indices

        self.conv1 = Conv3d(1, 8, 3, stride=2, padding=1,
                            batch_norm=batch_norm)
        self.conv2 = Conv3d(8, 8, 3, padding=1,
                            batch_norm=batch_norm)
        self.conv3 = Conv3d(8, 16, 3, padding=1,
                            batch_norm=batch_norm)
        self.maxpool4_l = nn.MaxPool3d(3,2,0,return_indices=True)
        self.conv4_r = Conv3d(16, 24, 3, stride=2,
                              batch_norm=batch_norm)
        self.concat4 = Merge(mode='concat')

        self.conv5_l_1 = Conv3d(40, 16, 1,
                                batch_norm=batch_norm)
        self.conv5_l_2 = Conv3d(16, 24, 3, padding=1,
                                batch_norm=batch_norm)

        self.conv5_r_1 = Conv3d(40, 16, 1,
                                batch_norm=batch_norm)
        self.conv5_r_2 = Conv3d(16, 16, (5,1,1), padding=(2,0,0),
                                batch_norm=batch_norm)
        self.conv5_r_3 = Conv3d(16, 16, (1,5,1), padding=(0,2,0),
                                batch_norm=batch_norm)
        self.conv5_r_4 = Conv3d(16, 16, (1,1,5), padding=(0,0,2),
                                batch_norm=batch_norm)
        self.conv5_r_5 = Conv3d(16, 24, 3, padding=1,
                                batch_norm=batch_norm)
        self.concat5 = Merge(mode='concat')

        self.conv6_l = Conv3d(48, 48, 3, stride=2,
                              batch_norm=batch_norm)
        self.maxpool6_r = nn.MaxPool3d(3,2,0,return_indices=True)
        self.concat6 = Merge(mode='concat')

    def forward(self, x):
        incs = []
        szs = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        szs += [x.shape]
        (xl,i),xr = self.maxpool4_l(x), self.conv4_r(x)
        incs += [i]
        x = self.concat4([xl,xr])

        xl, xr = self.conv5_l_1(x), self.conv5_r_1(x)
        xl, xr = self.conv5_l_2(xl), self.conv5_r_2(xr)
        xr = self.conv5_r_3(xr)
        xr = self.conv5_r_4(xr)
        xr = self.conv5_r_5(xr)
        x = self.concat5([xl,xr])

        szs += [x.shape]
        xl, (xr,i) = self.conv6_l(x), self.maxpool6_r(x)
        incs += [i]
        x = self.concat6([xl,xr])

        if self.return_indices:
            return x, incs, szs
        else:
            return x

class StemBlockTranspose(nn.Module):
    def __init__(self, batch_norm=True):
        super(StemBlockTranspose,self).__init__()
        self.split6 = Split([48])
        self.convT6_l = ConvTranspose3d(48, 48, 3, stride=2, output_padding=1,
                                        batch_norm=batch_norm)
        self.maxunpool6_r = nn.MaxUnpool3d(3, stride=2, padding=0)
        self.add6 = Merge(mode='add')

        self.split5 = Split([24])
        self.convT5_l_2 = ConvTranspose3d(24, 16, 3, padding=1,
                                         batch_norm=batch_norm)
        self.convT5_l_1 = ConvTranspose3d(16, 40, 1,
                                         batch_norm=batch_norm)

        self.convT5_r_5 = ConvTranspose3d(24, 16, 3, padding=1,
                                          batch_norm=batch_norm)
        self.convT5_r_4 = ConvTranspose3d(16, 16, (1,1,5), padding=(0,0,2),
                                          batch_norm=batch_norm)
        self.convT5_r_3 = ConvTranspose3d(16, 16, (1,5,1), padding=(0,2,0),
                                          batch_norm=batch_norm)
        self.convT5_r_2 = ConvTranspose3d(16, 16, (5,1,1), padding=(2,0,0),
                                          batch_norm=batch_norm)
        self.convT5_r_1 = ConvTranspose3d(16, 40, 1,
                                          batch_norm=batch_norm)

        self.add5 = Merge(mode='add')

        self.split4 = Split([16])
        self.maxunpool4_l = nn.MaxUnpool3d(3, stride=2, padding=0)
        self.convT4_r = ConvTranspose3d(24, 16, 3, stride=2,
                                        batch_norm=batch_norm)
        self.add4 = Merge(mode='add')

        self.convT3 = ConvTranspose3d(16, 8, 3, padding=1,
                                      batch_norm=batch_norm)
        self.convT2 = ConvTranspose3d(8, 8, 3, padding=1,
                                      batch_norm=batch_norm)
        self.convT1 = ConvTranspose3d(8, 1, 3, stride=2, padding=1,
                                      batch_norm=batch_norm)

    def forward(self, x, i, s):
        xl, xr = self.split6(x)
        xl, xr = self.convT6_l(xl), self.maxunpool6_r(xr, i[-1],
                                                      output_size=s[-1])
        x = self.add6([xl,xr])

        xl, xr = self.split5(x)
        xr = self.convT5_r_5(xr)
        xr = self.convT5_r_4(xr)
        xr = self.convT5_r_3(xr)
        xl, xr = self.convT5_l_2(xl), self.convT5_r_2(xr)
        xl, xr = self.convT5_l_1(xl), self.convT5_r_1(xr)
        x = self.add5([xl,xr])

        xl, xr = self.split4(x)
        xl, xr = self.maxunpool4_l(xl, i[-2]), self.convT4_r(xr)
        x = self.add4([xl,xr])

        x = self.convT3(x)
        x = self.convT2(x)
        x = self.convT1(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, batch_norm=True):
        super(InceptionA,self).__init__()
        self.conv1 = Conv3d(96, 8, 1, batch_norm=batch_norm)

        self.conv2_1 = Conv3d(96, 8, 1, batch_norm=batch_norm)
        self.conv2_2 = Conv3d(8, 8, 3, padding=1, batch_norm=batch_norm)

        self.conv3_1 = Conv3d(96, 8, 1, batch_norm=batch_norm)
        self.conv3_2 = Conv3d(8, 12, 3, padding=1, batch_norm=batch_norm)
        self.conv3_3 = Conv3d(12, 16, 3, padding=1, batch_norm=batch_norm)

        self.concat = Merge(mode='concat')

        self.conv_end = Conv3d(8+8+16, 96, 1, batch_norm=batch_norm)

    def forward(self, x):
        residual = x

        x1 = self.conv1(x)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)

        x = self.concat([x1,x2,x3])
        x = self.conv_end(x)

        x += residual

        return x

class InceptionATranspose(nn.Module):
    def __init__(self, batch_norm=True):
        super(InceptionATranspose,self).__init__()
        self.convT_top = ConvTranspose3d(96, 8+8+16, 1,
                                         batch_norm=batch_norm)
        self.split = Split([8,8])

        self.convT1 = ConvTranspose3d(8, 96, 1,
                                      batch_norm=batch_norm)

        self.convT2_2 = ConvTranspose3d(8, 8, 3, padding=1,
                                        batch_norm=batch_norm)
        self.convT2_1 = ConvTranspose3d(8, 96, 1,
                                        batch_norm=batch_norm)

        self.convT3_3 = ConvTranspose3d(16, 12, 3, padding=1,
                                        batch_norm=batch_norm)
        self.convT3_2 = ConvTranspose3d(12, 8, 3, padding=1,
                                        batch_norm=batch_norm)
        self.convT3_1 = ConvTranspose3d(8, 96, 1,
                                        batch_norm=batch_norm)
        self.add = Merge(mode='add')

    def forward(self, x):
        residual = x
        x = self.convT_top(x)
        x1, x2, x3 = self.split(x)

        x3 = self.convT3_3(x3)
        x3 = self.convT3_2(x3)
        x3 = self.convT3_1(x3)

        x2 = self.convT2_2(x2)
        x2 = self.convT2_1(x2)

        x1 = self.convT1(x1)

        x = self.add([residual,x1,x2,x3])

        return x

class ReductionA(nn.Module):
    def __init__(self, return_indices=False, batch_norm=True):
        super(ReductionA,self).__init__()
        self.return_indices = return_indices

        self.mp1 = nn.MaxPool3d(3,2,0, return_indices=True)

        self.conv2 = Conv3d(96, 96, 3, stride=2, batch_norm=batch_norm)

        self.conv3_1 = Conv3d(96, 64, 1, batch_norm=batch_norm)
        self.conv3_2 = Conv3d(64, 64, 3, padding=1, batch_norm=batch_norm)
        self.conv3_3 = Conv3d(64, 96, 3, stride=2, batch_norm=batch_norm)
        self.concat = Merge(mode='concat')

    def forward(self, x):
        sz = x.shape
        x1, i = self.mp1(x)

        x2 = self.conv2(x)

        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)

        x = self.concat([x1,x2,x3])

        if self.return_indices:
            return x, i, sz
        else:
            return x

class ReductionATranspose(nn.Module):
    def __init__(self, batch_norm=True):
        super(ReductionATranspose,self).__init__()
        self.split = Split([96,96])

        self.convT3_3 = ConvTranspose3d(96, 64, 3, stride=2,
                                        batch_norm=batch_norm)
        self.convT3_2 = ConvTranspose3d(64, 64, 3, padding=1,
                                        batch_norm=batch_norm)
        self.convT3_1 = ConvTranspose3d(64, 96, 1,
                                        batch_norm=batch_norm)

        self.convT2 = ConvTranspose3d(96, 96, 3, stride=2,
                                      batch_norm=batch_norm)

        self.mup1 = nn.MaxUnpool3d(3,stride=2,padding=0)

        self.add = Merge(mode='add')

    def forward(self, x, i, s):
        x1, x2, x3 = self.split(x)

        x3 = self.convT3_3(x3)
        x3 = self.convT3_2(x3)
        x3 = self.convT3_1(x3)

        x2 = self.convT2(x2)

        x1 = self.mup1(x1, i, output_size=s)

        x = self.add([x1,x2,x3])

        return x

class InceptionResNetCAE(_Base):
    def __init__(self, name='inception_resnet_cae', batch_norm=True):
        super(InceptionResNetCAE,self).__init__(name)
        self.stem = StemBlock(return_indices=True, batch_norm=batch_norm)

        self.inception_a = nn.Sequential(
            *[InceptionA(batch_norm=True) for _ in range(5)])
        self.reduction_a = ReductionA(return_indices=True,
                                      batch_norm=batch_norm)

        self.reduction_a_T = ReductionATranspose(batch_norm=batch_norm)
        self.inception_a_T = nn.Sequential(
            *[InceptionATranspose(batch_norm=batch_norm)\
              for _ in range(5)])

        self.stem_T = StemBlockTranspose(batch_norm=batch_norm)

    def forward(self, x):
        x,i1,s1 = self.stem(x)

        x = self.inception_a(x)
        x, i2, s2 = self.reduction_a(x)

        x = self.reduction_a_T(x, i2, s2)
        x = self.inception_a_T(x)

        x = self.stem_T(x,i1,s1)
        return x

class InceptionResNet(_Base):
    def __init__(self, num_classes,
                 name='inception_resnet',
                 batch_norm=True):
        super(InceptionResNet,self).__init__(name)
        self.stem = StemBlock(batch_norm=batch_norm)
        self.inception_a = nn.Sequential(
            *[InceptionA(batch_norm=batch_norm) for _ in range(5)])
        self.reduction_a = ReductionA(batch_norm=batch_norm)

        self.fconv1 = Conv3d(288, 1024, 1, batch_norm=batch_norm)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Dropout(p=0.2))

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)

        x = self.fconv1(x)

        x = self.avgpool(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return x

    def seperated_layers(self):
        return (nn.Sequential(self.stem, self.inception_a),
                nn.Sequential(self.fconv1, self.fc))

    def adapt(self, root, cae_name, running_k):
        print('Adapt {} wieght bias to {}'.format(cae_name, self.name))
        cae = InceptionResNetCAE(cae_name)
        cae.load_kfold(root, running_k, finalized=True)
        base_layer, _ = self.seperated_layers()

        for base_param, cae_param in zip(base_layer.parameters(),
                                         cae.parameters()):
            base_param.data.copy_(cae_param.data)
