import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule
from ..utils import xavier_init
from ..utils import Reorder
from mmcv.cnn import constant_init
import torch

class MeltFeature(nn.Module):
    def __init__(self, channel, num_ins):
        super(MeltFeature, self).__init__()
        self.channels_per_group = channel // num_ins
        self.num_ins = num_ins
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv1(x)
        # res = Reorder(res, self.num_ins)
        res = self.bn1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        # res = Reorder(res, self.channels_per_group)
        res = self.relu(res)
        return res

class FeatureReconfiguration(nn.Module):
    def __init__(self, inchannel, channel, scale, h, w, reduction=1, dim=1024):
        super(FeatureReconfiguration, self).__init__()
        self.channel = channel
        self.scale = scale
        self.conv1 = nn.Conv2d(inchannel, channel, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.h = h
        self.w = w

        self.spconv = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.fc = nn.Sequential(
                nn.Linear(self.h*self.w, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, self.h*self.w),
                nn.Sigmoid()
        )
        
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.donwscale = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                    nn.BatchNorm2d(channel))
            for i in range(scale)]
            )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        b, _, h, w = x.size()
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)

        y = self.spconv(res)
        y = y.view(b, h*w)
        y = self.fc(y).view(b, 1, h, w)
        y = res * y

        y = self.conv2(y)
        y = self.bn2(y)
        # y = self.relu(y)
        if self.scale != 0:
            y = self.donwscale(y)
        y = self.relu(y)
        return y

class DFPNv6(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=256,
                 num_outs=3,
                 in_scales=[8, 16, 32],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None,
                 return_inputs=False,
                 img_size=(512,512)):
        super(DFPNv6, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        self.in_scales = in_scales
        self.return_inputs = return_inputs
        self.img_size = img_size

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.dfr = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upscale_convs = nn.ModuleList()
        self.melt_channels = sum([mid_channels // 4**i for i in range(self.num_ins)])

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                mid_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                mid_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            if i != 0:
                upscale_conv = nn.PixelShuffle(upscale_factor=2**i)
                self.upscale_convs.append(upscale_conv)
            dfr = FeatureReconfiguration(self.melt_channels, mid_channels, i,
                                        self.img_size[0]//self.in_scales[0], self.img_size[1]//self.in_scales[0])

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.dfr.append(dfr)

        self.feat_melt = MeltFeature(self.melt_channels, self.num_ins)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                # in_channels = (self.in_channels[self.backbone_end_level - 1]
                #                if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        featurehierarchy = laterals[0]
        for i in range(1, used_backbone_levels):
            if self.in_scales[i] != self.in_scales[0]:
                featurehierarchy = torch.cat([featurehierarchy, 
                                            self.upscale_convs[i-1](laterals[i])], 1)
            else:
                featurehierarchy = torch.cat([featurehierarchy, laterals[i]], 1)

        meltfeat = self.feat_melt(featurehierarchy)
        # build se
        laterals = [
            dfr(meltfeat) + laterals[i]
            for i, dfr in enumerate(self.dfr)
        ]

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        if self.return_inputs:
            return (inputs, tuple(outs))
        else:
            return tuple(outs)
