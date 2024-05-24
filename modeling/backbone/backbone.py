from torch import nn
from collections import OrderedDict

from fcos.core.modeling.backbone import resnet, fpn as fpn_module
from fcos.core.modeling.make_layers import conv_with_kaiming_uniform
from fcos.core.modeling import registry

from fcos.core.modeling.backbone import resnet
from fcos.core.modeling.backbone.resnet import _STAGE_SPECS

_STAGE_SPECS.register("R-50-FPN-HYBRID", resnet.ResNet50FPNStagesTo5)
_STAGE_SPECS.register("R-101-FPN-HYBRID", resnet.ResNet101FPNStagesTo5)

from fcos.core.utils.c2_model_loading import C2_FORMAT_LOADER, _C2_STAGE_NAMES
from fcos.core.utils.c2_model_loading import _load_c2_pickled_weights, _rename_weights_for_resnet, _rename_conv_weights_for_deformable_conv_layers

def load_resnet_hybrid_c2_format(cfg, f):
    state_dict = _load_c2_pickled_weights(f)
    conv_body = cfg.MODEL.BACKBONE.CONV_BODY
    arch = conv_body.replace("-C4", "").replace("-C5", "").replace("-FPN", "").replace("-HYBRID", "")
    stages = _C2_STAGE_NAMES[arch]
    state_dict = _rename_weights_for_resnet(state_dict, stages)
    state_dict = _rename_conv_weights_for_deformable_conv_layers(state_dict, cfg)
    return dict(model=state_dict)

C2_FORMAT_LOADER.register("R-50-FPN-HYBRID", load_resnet_hybrid_c2_format)
C2_FORMAT_LOADER.register("R-101-FPN-HYBRID", load_resnet_hybrid_c2_format)


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