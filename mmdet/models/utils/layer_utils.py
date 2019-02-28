import torch
import warnings

import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init

from .norm import build_norm_layer

def Reorder(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class IGCv3_block(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(IGCv3_block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio,kernel_size = 1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            #permutation
            PermutationBlock(groups=2),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size =3, stride= stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size =1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(oup),
            # permutation
            PermutationBlock(groups= int(round((oup/2)))),
        )

    def forward(self, x):
      return self.conv(x)

class Class_predict(nn.Module):
    def __init__(self, feat_channels, num_anchors, cls_out_channels, bn=True):
        super(Class_predict, self).__init__()
        self.num_anchors = num_anchors
        self.cls_out_channels = cls_out_channels
        self.bn = bn
        self.feat_channels = feat_channels
        self.conv1 = nn.Conv2d(
                        feat_channels,
                        self.num_anchors * self.cls_out_channels,
                        (1, 3),
                        stride=1,
                        padding=(0, 1),
                        groups=self.cls_out_channels)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(self.num_anchors * self.cls_out_channels)
        self.conv2 = nn.Conv2d(
                        self.num_anchors * self.cls_out_channels,
                        self.num_anchors * self.cls_out_channels,
                        (3, 1),
                        stride=1,
                        padding=(1, 0),
                        groups=self.cls_out_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.bn:
            y = self.bn1(y)
        y = Reorder(y, self.cls_out_channels)
        y = self.conv2(y)
        return y