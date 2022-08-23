# Copyright (c) OpenMMLab. All rights reserved.
# Modified from csuhan: https://github.com/csuhan/ReDet
import warnings

import math
import e2cnn.nn as enn
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

from e2cnn import gspaces
from ..builder import ROTATED_NECKS

def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = False):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= math.sqrt(N)

    planes = planes / N
    planes = int(planes)

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = False):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


def conv3x3(gspace, inplanes, out_planes, stride=1, padding=1, dilation=1, bias=False, fixparams=False):
    """3x3 convolution with padding"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, out_planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)


def conv1x1(gspace, inplanes, out_planes, stride=1, padding=0, dilation=1, bias=False, fixparams=False):
    """1x1 convolution"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, out_planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)

def build_norm_layer(gspace, num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)

class ConvModule(enn.EquivariantModule):
    """ConvModule.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): The size of kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        activation (str, optional): Activation layer in ConvModule.
            Default: None.
        inplace (bool): can optionally do the operation in-place.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=False,
                 gspace=None,
                 fixparams=False,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.in_type = FIELD_TYPE['regular'](
            gspace, in_channels, fixparams=fixparams)
        self.out_type = FIELD_TYPE['regular'](
            gspace, out_channels, fixparams=fixparams)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        # build convolution layer
        if kernel_size == 1:
            self.conv = conv1x1(
                            gspace,
                            in_channels,
                            out_channels,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=False,
                            fixparams=fixparams)
        elif kernel_size == 3:
            self.conv = conv3x3(
                            gspace,
                            in_channels,
                            out_channels,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            bias=False,
                            fixparams=fixparams)
        else:
            NotImplementedError

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = padding
        self.groups = groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if conv_cfg is not None and conv_cfg['type'] == 'ORConv':
                norm_channels = int(norm_channels * 8)
            self.norm_name, norm = build_norm_layer(gspace, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError(
                    f'{self.activation} is currently not supported.')
            if self.activation == 'relu':
                self.activate = enn.ReLU(self.conv.out_type, inplace=True)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        """Get normalizion layer's name."""
        return getattr(self, self.norm_name)

    def init_weights(self):
        """Initialize weights of the head."""
        nonlinearity = 'relu' if self.activation is None \
            else self.activation  # noqa: F841

    def forward(self, x, activate=True, norm=True):
        """Forward function of ConvModule."""
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x

    def evaluate_output_shape(self, input_shape):
        """Evaluate output shape."""
        return input_shape


@ROTATED_NECKS.register_module()
class DeformReFPN(BaseModule):
    """ReFPN.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int, optional): Index of the start input backbone level
            used to build the feature pyramid. Default: 0.
        end_level (int, optional): Index of the end input backbone level
            (exclusive) to build the feature pyramid. Default: -1, which means
            the last level.
        add_extra_convs (bool, optional): It decides whether to add conv layers
            on top of the original feature maps. Default to False.
        extra_convs_on_inputs (bool, optional): It specifies the source feature
            map of the extra convs is the last feat map of neck inputs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        activation (str, optional): Activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 orientation=8,
                 flip = False,
                 fixparams=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):

        super(DeformReFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.orientation = orientation
        self.fixparams = fixparams
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        if flip == False:
            self.gspace = gspaces.Rot2dOnR2(orientation)
        else:
            self.gspace = gspaces.FlipRot2dOnR2(orientation)

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
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                gspace=self.gspace,
                inplace=False)
            up_sample = enn.R2Upsampling(l_conv.out_type, 2, mode='nearest', align_corners=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                gspace=self.gspace,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.up_samples.append(up_sample)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    gspace=self.gspace,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.max_pools = nn.ModuleList()
        self.relus = nn.ModuleList()

        used_backbone_levels = len(self.lateral_convs)
        if self.num_outs > used_backbone_levels:
            # use max pool to get more levels on top of outputs
            # (e.g., Rotated Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    self.max_pools.append(
                        enn.PointwiseMaxPool(self.fpn_convs[-1].out_type, kernel_size=1, stride=2, padding=0))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                for i in range(used_backbone_levels + 1, self.num_outs):
                    self.relus.append(enn.ReLU(self.fpn_convs[-1].out_type, inplace=True))

    @auto_fp16()
    def forward(self, inputs):
        """Forward function of ReFPN."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += self.up_samples[i](laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Rotated Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(self.max_pools[i](outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](self.relus[i](outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # convert to tensor
        outs = [out.tensor for out in outs]

        return tuple(outs)
