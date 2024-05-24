import torch
from torch import nn
from collections import OrderedDict

from fcos.core.modeling.backbone import fpn as fpn_module
from fcos.core.modeling.make_layers import conv_with_kaiming_uniform

class FPN:
    def __init__(self, cfg, backbone, fpn_type="original"):
        self.cfg = cfg
        self.backbone = backbone
        self.fpn_type = fpn_type

    def build(self):
        # Assuming the model outputs feature maps at indices 0, 1, 2, 3
        in_channels_list = [f['num_chs'] for f in self.backbone.feature_info]

        if self.fpn_type == "retinanet":
            in_channels_list[0] = 0
            top_blocks = fpn_module.LastLevelP6P7(in_channels_list[-1], self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS)
        elif self.fpn_type == "hybrid":
            top_blocks = fpn_module.LastLevelP6P7(in_channels_list[-1], self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS)
        else:  # Default to original FPN
            top_blocks = fpn_module.LastLevelMaxPool()

        fpn = fpn_module.FPN(
            in_channels_list=in_channels_list,
            out_channels=self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
            conv_block=conv_with_kaiming_uniform(self.cfg.MODEL.FPN.USE_GN, self.cfg.MODEL.FPN.USE_RELU),
            top_blocks=top_blocks
        )

        model = nn.Sequential(OrderedDict([
            ("body", self.backbone),
            ("fpn", fpn)
        ]))
        model.out_channels = self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

        return model