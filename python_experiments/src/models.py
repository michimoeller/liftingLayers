"""Model Definitions."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from pytorch_tools.models import BaseNN, init_weights_kaiming


class LiftingLayerMultiD(nn.Module):

    def __init__(self):
        super(LiftingLayerMultiD, self).__init__()

    def forward(self, x):
        x = torch.cat([F.relu(x), -1.0 * F.relu(-1.0 * x)], dim=1)
        return x


class LiftNetBlock(nn.Module):

    def __init__(self, in_channels, features):
        super(LiftNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, features, (3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(features),
            LiftingLayerMultiD(),
            nn.Conv2d(2 * features, features, (3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(features),
            LiftingLayerMultiD(), )

    def forward(self, x):
        x = self.layer(x)
        return x


class LiftNet(BaseNN):

    def __init__(self, channels, features, kaiming_init=False):
        super(LiftNet, self).__init__()
        self._kaiming_init = kaiming_init
        self.first_conv = LiftNetBlock(channels, features)
        self.center_convs = nn.Sequential(*[LiftNetBlock(2 * features, features) for _ in range(7)])
        self.last_conv = nn.Conv2d(2 * features, 1, (3, 3), padding=1, stride=1)
        # self.center_convs = nn.Sequential(*[LiftNetBlock(2 * features, features) for _ in range(6)])
        # self.last_conv = nn.Sequential(
        #     nn.Conv2d(2 * features, 64, (3, 3), padding=1, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, (3, 3), padding=1, stride=1), )

        self.init_parameters()

    def init_parameters(self):
        if self._kaiming_init:
            self.apply(init_weights_kaiming)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.center_convs(x)
        x = self.last_conv(x)
        return x


class DNCNNBlock(nn.Module):

    def __init__(self, features):
        super(DNCNNBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(features, features, (3, 3), padding=1, stride=1, bias=False),
            # TODO: check momentum in Matlab
            nn.BatchNorm2d(features),
            nn.ReLU(),)

    def forward(self, x):
        x = self.layer(x)
        return x


class DNCNN(BaseNN):

    def __init__(self, channels, num_blocks=15, features=64, kaiming_init=False):
        # TODO: Xavier init
        super(DNCNN, self).__init__()
        self._kaiming_init = kaiming_init
        self.first_conv = nn.Sequential(
            nn.Conv2d(channels, features, (3, 3), padding=1, stride=1, bias=False),
            nn.ReLU(), )

        self.center_convs = nn.Sequential(*[DNCNNBlock(features) for _ in range(num_blocks)])

        # TODO: check WeightLearnRateFactor from Matlab
        self.last_conv = nn.Conv2d(features, 1, (3, 3), padding=1, stride=1, bias=False)

        self.init_parameters()

    def init_parameters(self):
        if self._kaiming_init:
            self.apply(init_weights_kaiming)
        # for _, module in self.named_modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.running_var *= 0.01

    def forward(self, x):
        x = self.first_conv(x)
        x = self.center_convs(x)
        x = self.last_conv(x)
        return x
