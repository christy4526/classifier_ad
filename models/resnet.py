from __future__ import absolute_import, division, print_function

import os
from os.path import join as pjoin
from collections import OrderedDict as odict

import torch
import torch.nn as nn
import torch.nn.parallel.data_parallel as dp
import numpy as np

from .base import Conv3d, ConvTranspose3d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 batch_norm=True, activation=nn.ReLU,
                 weights_initializer=None):
        super(BasicBlock, self).__init__()
        bias = not batch_norm
        self.conv1 = Conv3d(inplanes, planes, 3, stride, 1, bias=bias,
                            batch_norm=batch_norm, activation=activation,
                            weights_initializer=weights_initializer)
        self.conv2 = Conv3d(planes, planes, 3, 1, 1, bias=bias,
                            batch_norm=batch_norm, activation=None,
                            weights_initializer=weights_initializer)
        self.downsample = downsample
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


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 batch_norm=True, activation=nn.ReLU,
                 weights_initializer=None):
        super(Bottleneck, self).__init__()
        bias = not batch_norm
        self.conv1 = Conv3d(inplanes, planes, 1, 1, bias=bias,
                            batch_norm=batch_norm, activation=activation,
                            weights_initializer=weights_initializer)
        self.conv2 = Conv3d(planes, planes, 3, stride, 1, bias=bias,
                            batch_norm=batch_norm, activation=activation,
                            weights_initializer=weights_initializer)
        self.conv3 = Conv3d(planes, planes*self.expansion, 1, 1, bias=bias,
                            batch_norm=batch_norm, activation=None,
                            weights_initializer=weights_initializer)
        self.downsample = downsample
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


class ResNet(nn.Module):
    def __init__(self, num_classes, name, block, layers,
                 batch_norm=True, activation=nn.ReLU,
                 weights_initializer=None):
        self.inplanes = 16
        super(ResNet, self).__init__(name)

        self.batch_norm = batch_norm
        self.bias = not batch_norm
        self.activation = activation
        self.weights_initializer = weights_initializer

        self.stem = nn.Sequential(
            Conv3d(1, 16, 5, stride=2, padding=2, bias=self.bias,
                   batch_norm=batch_norm, activation=activation,
                   weights_initializer=weights_initializer),
            nn.MaxPool3d(2, 2))

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = nn.Sequential(
            self._make_layer(block, 64, layers[3]),
            Conv3d(64*block.expansion, 128*block.expansion, 1,
                   batch_norm=batch_norm, activation=activation,
                   bias=False, weights_initializer=weights_initializer))

        self.avgpool = nn.AvgPool3d((7, 9, 7), stride=1)
        self.fc = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv3d(
                self.inplanes, planes*block.expansion, 1, stride,
                bias=self.bias, batch_norm=self.batch_norm, activation=None,
                weights_initializer=self.weights_initializer)

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        batch_norm=self.batch_norm, activation=self.activation,
                        weights_initializer=self.weights_initializer)]
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes,
                             batch_norm=self.batch_norm,
                             activation=self.activation,
                             weights_initializer=self.weights_initializer)]
        return nn.Sequential(*layers)

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


def resnet11(num_classes, name='resnet11', cae=False, **kwargs):
    model = ResNet(num_classes, name, BasicBlock, [1, 1, 1, 1])
    return model


def resnet19(num_classes, name='resnet19', **kwargs):
    model = ResNet(num_classes, name, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet35(num_classes, name='resnet35', **kwargs):
    model = ResNet(num_classes, name, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet51(num_classes, name='resnet51', **kwargs):
    model = ResNet(num_classes, name, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(num_classes, name='resnet101', **kwargs):
    model = ResNet(num_classes, name, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
