from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .base import _Base
from .base import Conv3d, ConvTranspose3d, SplitConv3d
from .base import DropoutLinear
from .base import Merge, Split

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

class Reduction(Block):
    def __init__(self, in_channels, bottle_channels, tower2_out_channels,
                 return_indices=False, batch_norm=False,
                 activation=nn.SELU):
        super(Reduction, self).__init__()
        self.return_indices = return_indices

        self.tower1 = nn.MaxPool3d(3,2,0, return_indices=return_indices)
        self.tower2 = nn.Sequential(
            Conv3d(in_channels, bottle_channels, 1,
                   batch_norm=batch_norm, activation=activation),
            SplitConv3d(bottle_channels, tower2_out_channels, 3, stride=2,
                        batch_norm=batch_norm, activation=activation))

        self.concat = Merge(mode='concat')

    def forward(self ,x):
        s = x.shape
        if self.return_indices:
            tower1_x, i = self.tower1(x)
        else:
            tower1_x = self.tower1(x)

        tower2_x = self.tower2(x)
        x = self.concat([tower1_x, tower2_x])

        if self.return_indices:
            return x, i, s
        return x


class Stem(Block):
    def __init__(self, return_indices=False, batch_norm=False,
                 activation=nn.SELU):
        super(Stem, self).__init__()
        self.return_indices = return_indices

        self.conv1 = Conv3d(1, 16, 3, stride=2,
                            batch_norm=batch_norm, activation=activation)
        self.conv2 = Conv3d(16, 16, 3, padding=1,
                            batch_norm=batch_norm, activation=activation)
        self.conv3 = Conv3d(16, 24, 3,
                            batch_norm=batch_norm, activation=activation)
        self.reduction3 = Reduction(24, 8, 32, batch_norm=batch_norm,
                                    activation=activation,
                                    return_indices=return_indices)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.reduction3(x)

class InceptionA(Block):
    def __init__(self, batch_norm=False, activation=nn.SELU):
        super(InceptionA, self).__init__()
        in_channels = 56
        bottle_channels = 8
        t3_out_channels = bottle_channels*2
        t3_mid_channels = int((bottle_channels+t3_out_channels)/2)

        self.tower1 = Conv3d(in_channels, bottle_channels, 1,
                             batch_norm=batch_norm, activation=activation)
        self.tower2 = nn.Sequential(
            Conv3d(in_channels, bottle_channels, 1,
                   batch_norm=batch_norm, activation=activation),
            Conv3d(bottle_channels, bottle_channels, 3,
                   batch_norm=batch_norm, activation=activation))

        self.tower3 = nn.Sequential(
            Conv3d(in_channels, bottle_channels, 1,
                   batch_norm=batch_norm, activation=activation),
            Conv3d(bottle_channels, t3_mid_channels,3,
                   batch_norm=batch_norm, activation=activation),
            Conv3d(t3_mid_channels, t3_out_channels, 3,
                   batch_norm=batch_norm, activation=activation))

        self.conv = Conv3d(2*bottle_channels+t3_out_channels, in_channels, 1,
                           batch_norm=batch_norm, activation=None)
        self.activation = activation(inplace=True)

        self.concat = Merge(mode='concat')

    def forward(self, x):
        residual = x
        tower1_x = self.tower1(x)
        tower2_x = self.tower2(x)
        tower3_x = self.tower3(x)
        x = self.concat([tower1_x, tower2_x, tower3_x])

        x = self.conv(x)
        x = self.activation(x+residual)
        return x

class InceptionB(Block):
    def __init__(self, batch_norm=False, activation=nn.SELU):
        super(InceptionB, self).__init__()
        in_channels = 120
        bottle_channels = 16
        t2_out_channels = 28

        self.tower1 = Conv3d(in_channels, bottle_channels, 1,
                             batch_norm=batch_norm, activation=activation)
        self.tower2 = nn.Sequential(
            Conv3d(in_channels, bottle_channels, 1,
                   batch_norm=batch_norm, activation=activation),
            SplitConv3d(bottle_channels, t2_out_channels, 5, padding=2,
                        batch_norm=batch_norm, activation=activation))
        self.concat = Merge(mode='concat')
        self.conv = Conv3d(2*t2_out_channels, 1,
                           batch_norm=batch_norm, activation=None)
        self.activation = activation(inplace=True)

    def forward(self, x):
        residual = x
        tower1_x = self.tower1(x)
        tower2_x = self.tower2(x)
        x = self.concat([tower1_x, tower2_x])

        x = self.conv(x)
        x = self.activation(x+residual)
        return x

class InceptionC(Block):
    def __init__(self, batch_norm=False, activation=nn.SELU):
        super(InceptionC, self).__init__()

class InceptionResNetSmallCAE(_Base):
    def __init__(self, name='inception_resnet_small_cae',
                 batch_norm=False, activation=nn.SELU):
        super(InceptionResNetSmallCAE, self).__init__(name)
        raise NotImplementedError


class InceptionResNetSmall(_Base):
    def __init__(self, num_classes, name='inception_resnet_small',
                 batch_norm=False, activation=nn.SELU):
        super(InceptionResNetSmall, self).__init__(name)

        self.stem_block = Stem(return_indices=False, batch_norm=batch_norm,
                               activation=activation)

        self.conv_block1 = nn.Sequential(
            *[InceptionA(batch_norm=batch_norm, activation=activation)\
              for _ in range(3)])

        self.reduction1 = Reduction(56, 16, 64, return_indices=False,
                                    batch_norm=batch_norm,
                                    activation=activation)

        self.conv_block2 = nn.Sequential(
            *[InceptionB(batch_norm=batch_norm, activation=activation)\
              for _ in range(5)])

        self.reduction2 = Reduction(120, 32, 96, return_indices=False,
                                    batch_norm=batch_norm,
                                    activation=activation)

        self.conv_block3 = InceptionC(batch_norm=batch_norm,
                                      activation=activation)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.AlphaDropout() if activation==nn.SELU else nn.Dropout3d(),
            nn.Linear(192, num_classes),
            activation(inplace=True))

    def seperated_layers(self):
        return (nn.Sequential(self.stem_block,self.conv_block1,self.reduction1,
                              self.conv_block2,self.reduction2,self.conv_block3),
                self.fc)

    def forward(self, x):
        x = self.stem_block(x)

        x = self.conv_block1(x)
        x = self.reduction1(x)

        x = self.conv_block2(x)
        x = self.reduction2(x)

        x = self.conv_block3(x)
        x = self.avgpool(x)

        x = x.view(-1, 192)
        x = self.fc(x)
        return x
