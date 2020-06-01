from __future__ import absolute_import, division, print_function

import os
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.parallel.data_parallel as dp
import numpy as np

from .base import _Base, Conv3d, ConvTranspose3d


class ResNeXtBlock(nn.Module):
    def __init__(self):
        super(ResNeXtBlock,self).__init__()
    def _make_downsample(self, in_channels, out_channels, stride,
                         batch_norm=True, weights_initializer=None):
        if in_channels!=out_channels:
            self.downsample = Conv3d(in_channels, out_channels, 1, stride,
                                     bias=not batch_norm,
                                     weights_initializer=weights_initializer,
                                     batch_norm=batch_norm, activation=None)
        else:
            self.downsample = None

class BaseBlock(ResNeXtBlock):
    def __init__(self, in_channels, out_channels, cardinality,
                 stride=1, batch_norm=True, activation=nn.ReLU,
                 weights_initializer=None):
        super(BaseBlock,self).__init__()
        bias = not batch_norm
        self.conv1 = Conv3d(in_channels, out_channels, 3, stride, padding=1,
                            groups=cardinality, bias=bias,
                            batch_norm=batch_norm, activation=activation,
                            weights_initializer=weights_initializer)
        self.conv2 = Conv3d(out_channels, out_channels, 3, padding=1,
                            groups=cardinality, bias=bias,
                            batch_norm=batch_norm, activation=None,
                            weights_initializer=weights_initializer)
        self._make_downsample(in_channels, out_channels, stride,
                              batch_norm=batch_norm,
                              weights_initializer=weights_initializer)
        self.activation = activation(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out

class Bottleneck(ResNeXtBlock):
    def __init__(self, in_channels, channels, out_channels, cardinality,
                 stride=1, batch_norm=True, activation=nn.ReLU,
                 weights_initializer=None):
        super(Bottleneck,self).__init__()
        bias = not batch_norm
        self.conv1 = Conv3d(in_channels, channels, 1, bias=bias,
                            batch_norm=batch_norm, activation=activation,
                            weights_initializer=weights_initializer)
        self.conv2 = Conv3d(channels, channels, 3, stride, padding=1,
                            groups=cardinality, bias=bias,
                            batch_norm=batch_norm, activation=activation,
                            weights_initializer=weights_initializer)
        self.conv3 = Conv3d(channels, out_channels, 1, bias=bias,
                            batch_norm=batch_norm, activation=None,
                            weights_initializer=weights_initializer)
        self._make_downsample(in_channels, out_channels, stride,
                              batch_norm=batch_norm,
                              weights_initializer=weights_initializer)
        self.activation = activation(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out

class ResNeXt(_Base):
    def __init__(self, num_classes, name='resnext',
                 batch_norm=True, activation=nn.ReLU,
                 weights_initializer=None):
        super(ResNeXt, self).__init__(name)

        bias = not batch_norm
        self.stem = nn.Sequential(
            Conv3d(1, 16, 5, stride=2, padding=2,
                   bias=bias, batch_norm=batch_norm, activation=activation,
                   weights_initializer=weights_initializer))
        self.layer1 = nn.Sequential(
            Bottleneck(16, 32, 64, 8, stride=2,
                      batch_norm=batch_norm, activation=activation,
                      weights_initializer=weights_initializer),
            Bottleneck(64, 32, 64, 8,
                      batch_norm=batch_norm, activation=activation,
                      weights_initializer=weights_initializer),
            Bottleneck(64, 32, 64, 8,
                      batch_norm=batch_norm, activation=activation,
                      weights_initializer=weights_initializer))
        self.layer2 = nn.Sequential(
            Bottleneck(64, 64, 128, 16, stride=2,
                       batch_norm=batch_norm, activation=activation,
                       weights_initializer=weights_initializer),
            Bottleneck(128, 64, 128, 16,
                       batch_norm=batch_norm, activation=activation,
                       weights_initializer=weights_initializer),
            Bottleneck(128, 64, 128, 16,
                       batch_norm=batch_norm, activation=activation,
                       weights_initializer=weights_initializer),
            Bottleneck(128, 64, 128, 16,
                       batch_norm=batch_norm, activation=activation,
                       weights_initializer=weights_initializer))
        self.layer3 = nn.Sequential(
            Bottleneck(128, 128, 256, 32, stride=2,
                       batch_norm=batch_norm, activation=activation,
                       weights_initializer=weights_initializer),
            Bottleneck(256, 128, 256, 32,
                       batch_norm=batch_norm, activation=activation,
                       weights_initializer=weights_initializer))
        self.pool = nn.Sequential(
            nn.AvgPool3d((7,9,7)))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

    def seperated_layers(self):
        return (nn.Sequential(self.stem, self.layer1, self.layer2, self.layer3),
                nn.Sequential(self.pool, self.fc))
