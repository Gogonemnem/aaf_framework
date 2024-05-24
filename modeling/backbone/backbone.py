from torch import nn
from collections import OrderedDict

from fcos.core.modeling.backbone import resnet, fpn as fpn_module
from fcos.core.modeling.make_layers import conv_with_kaiming_uniform
from fcos.core.modeling import registry

@registry.BACKBONES.register("R-50-FPN-HYBRID")
@registry.BACKBONES.register("R-101-FPN-HYBRID")
def build_resnet_fpn_p2p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN,
                                             cfg.MODEL.FPN.USE_RELU),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model