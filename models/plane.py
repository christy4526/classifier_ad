from __future__ import absolute_import, division, print_function

from collections import OrderedDict as odict

import os
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from .base import _Base, Conv3d, ConvTranspose3d
from .base import DropoutLinear


class Plane(_Base):
    def __init__(self, num_classes, name='plane',
                 weights_initializer=None):
        super(Plane, self).__init__(name)

        self.stem = nn.Sequential(
            Conv3d(1, 16, 5, stride=2, padding=2, bias=False, batch_norm=True,
                   weights_initializer=weights_initializer),
            nn.MaxPool3d(2, 2),
        )
        self.layer1 = nn.Sequential(
            Conv3d(16, 16, 3, 1, 1, bias=False, batch_norm=True,
                   weights_initializer=weights_initializer)
        )
        self.layer2 = nn.Sequential(
            Conv3d(16, 32, 3, 1, 1, bias=False, batch_norm=True,
                   weights_initializer=weights_initializer),
            nn.MaxPool3d(2, 2)
        )
        self.layer3 = nn.Sequential(
            Conv3d(32, 64, 3, 1, 1, bias=False, batch_norm=True,
                   weights_initializer=weights_initializer),
            nn.MaxPool3d(2, 2)
        )
        self.layer4 = nn.Sequential(
            Conv3d(64, 64, 3, 1, 1, bias=False, batch_norm=True,
                   weights_initializer=weights_initializer),
            Conv3d(64, 128, 1, 1, 0, bias=False, batch_norm=True,
                   weights_initializer=weights_initializer)
        )
        self.avgpool = nn.AvgPool3d((7, 9, 7), stride=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()
        self.stem = nn.Sequential(
            Conv3d(1, 16, 5, stride=2, padding=2, bias=False, batch_norm=True),
            nn.MaxPool3d(3, 2, 1, return_indices=True))

        self.layer1 = nn.Sequential(
            Conv3d(16, 16, 3, 1, 1, bias=False, batch_norm=True))

        self.layer2 = nn.Sequential(
            Conv3d(16, 32, 3, 1, 1, bias=False, batch_norm=True),
            nn.MaxPool3d(2, 2, return_indices=True))

        self.layer3 = nn.Sequential(
            Conv3d(32, 64, 3, 1, 1, bias=False, batch_norm=True),
            nn.MaxPool3d(2, 2, return_indices=True))

    def forward(self, x):
        i = []
        s = []
        s += [x.shape]
        x = self.stem[0](x)
        s += [x.shape]
        x, i0 = self.stem[1](x)
        i += [i0]
        s += [x.shape]
        x = self.layer1(x)
        s += [x.shape]
        x, i2 = self.layer2(x)
        i += [i2]
        s += [x.shape]
        x, i3 = self.layer3(x)
        i += [i3]

        return x, i, s


class _Decoder(nn.Module):
    def __init__(self):
        super(_Decoder, self).__init__()
        self.layer3 = nn.Sequential(
            nn.MaxUnpool3d(2, 2),
            Conv3d(64, 32, 3, 1, 1, bias=False, batch_norm=True))

        self.layer2 = nn.Sequential(
            nn.MaxUnpool3d(2, 2),
            Conv3d(32, 16, 3, 1, 1, bias=False, batch_norm=True))

        self.layer1 = nn.Sequential(
            Conv3d(16, 16, 3, 1, 1, bias=False, batch_norm=True))

        self.stem = nn.Sequential(
            nn.MaxUnpool3d(3, 2, 1),
            ConvTranspose3d(16, 1, 5, 2, 2, bias=False, batch_norm=True,
                            output_padding=1))

    def forward(self, x, i, s):
        x = self.layer3[0](x, i[-1])
        x = self.layer3[1](x)
        x = self.layer2[0](x, i[-2])
        x = self.layer2[1](x)
        x = self.layer1(x)
        x = self.stem[0](x, i[0], output_size=s[1])
        x = self.stem[1](x)

        return x


class PlaneCAE(_Base):
    def __init__(self, name='plane_cae', num_decoder=1):
        super(PlaneCAE, self).__init__(name)

        self.encoder = _Encoder()

        self.num_decoder = num_decoder
        self.decoder = []
        for _ in range(num_decoder):
            self.decoder += [_Decoder()]
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x, i, s = self.encoder(x)
        xs = []
        for decoder_idx in range(self.num_decoder):
            xs += [self.decoder[decoder_idx](x, i, s)]
        return torch.cat(xs, dim=1)
