import math

import numpy as np
from torch import nn
import torch
import cv2
import tarfile
import os
import glob
from PIL import Image
import torch.nn.functional as F


# lets first explain some logic:
#     - we want to upsacle a low res image to a higher res image
#     - we dont want to lose geometric dataclasses
#     - we want to make it fast ENOUGHT
#
# so...
#  - we use resnet blocks (conv -> relu -> conv)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        # print(self.layers(x).shape)
        return x + self.layers(x) * self.res_scale


def conv(features, inputs, kernel_size=3, atcn=True):
    layers = [nn.Conv2d(inputs, features, kernel_size, padding=kernel_size // 2)]
    if atcn:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)
    # resnet blocks are conv->relu->conv, this func sets it up easily


def res_block(nf):
    return ResSequential(
        [conv(nf, nf), conv(nf, nf, atcn=False)],
        0.1) # conv-> Relu-> conv


# basic resnet block


def upsample(inputs, features, scale=10):
    layers = []
    for i in range(int(math.log(scale, 2))):
        layers += [conv(features * 4, inputs), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


# upsampled block

class deCNN(nn.Module):
    def __init__(self, scale, features=64, hidden_layers=20):
        super(deCNN, self).__init__()
        layers = []
        # initial conv layer, 3 channels for H, S, V
        self.scale = scale

        # self.first = layers.append(conv(features, 3))
        self.first = conv(features, 3)
        for i in range(hidden_layers): layers.append(res_block(features))
        layers.append(conv(features, features))
        layers.append(upsample(features, features, scale))
        layers.append(nn.BatchNorm2d(features))
        # layers.append(conv(3, features, False))
        self.layers = nn.Sequential(*layers)
        self.last = conv(3, 64, kernel_size=3, atcn=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.first(x)
        x = self.layers(x)
        x = self.last(x)
        return x


class SuperResolutionLoss(nn.Module):
    def __init__(self):
        super(SuperResolutionLoss, self).__init__()

    def forward(self, output, target):
        target_scaled = F.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=False)
        loss = F.huber_loss(output, target_scaled)

        return loss