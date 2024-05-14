import torch
from torch import nn
from collections import OrderedDict

from timm import create_model

from fcos.core.modeling.backbone import fpn as fpn_module
from fcos.core.modeling import registry
from fcos.core.modeling.make_layers import conv_with_kaiming_uniform

def _freeze_backbone(self, freeze_at):
    # Always freeze the patch embeddings
    for param in self.patch_embed.parameters():
        param.requires_grad = False
    
    # Freeze the specified stages
    for i in range(freeze_at):
        stage = getattr(self, f'stages_{i}', None)
        if stage is not None:
            for param in stage.parameters():
                param.requires_grad = False

def build_pvtv2_fpn_backbone(cfg, variant):
    # Create the PVT model using TIMM
    body = create_model(
        variant,
        pretrained=cfg.PRETRAINED_BACKBONE,
        features_only=True,
        out_indices=(0, 1, 2, 3)
    )

    # Dynamically add the _freeze_backbone method to the body object to conform to fcos package
    setattr(body, '_freeze_backbone', _freeze_backbone.__get__(body, type(body)))


    # Assuming PVTv2 outputs feature maps at indices 0, 1, 2, 3
    in_channels_list = [f['num_chs'] for f in body.feature_info]
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
        conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
        top_blocks=fpn_module.LastLevelMaxPool()
    )

    model = nn.Sequential(OrderedDict([
        ("body", body),
        ("fpn", fpn)
    ]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    
    return model

# Create a dictionary to map the uppercase variant names to the lowercase ones used in TIMM
variant_name_mapping = {
    'PVT-V2-B0': 'pvt_v2_b0',
    'PVT-V2-B1': 'pvt_v2_b1',
    'PVT-V2-B2': 'pvt_v2_b2',
    'PVT-V2-B3': 'pvt_v2_b3',
    'PVT-V2-B4': 'pvt_v2_b4',
    'PVT-V2-B5': 'pvt_v2_b5',
    'PVT-V2-B2-LI': 'pvt_v2_b2_li'
}

# Register all PVTv2 backbones
for variant in variant_name_mapping.keys():
    # Get the corresponding lowercase variant name
    lowercase_variant = variant_name_mapping[variant]
    # Note: The use of a lambda function here to avoid closure over the variant variable in the loop
    backbone_builder = lambda cfg, variant=lowercase_variant: build_pvtv2_fpn_backbone(cfg, variant)
    registry.BACKBONES.register(variant)(backbone_builder)

# from fcos.core.modeling.backbone.pvtv2 import PyramidVisionTransformerV2
# def build_pvtv2_fpn_backbone(cfg, variant, pretrained=True):
#     # Here, 'variant' would be a dictionary with the specific configuration of the PVTv2 model you wish to use.
#     body = PyramidVisionTransformerV2(**variant)

#     if pretrained:
#         # Load the pretrained weights, if available.
#         # You would need to implement this part according to how you manage checkpoints.
#         checkpoint = torch.load(cfg.MODEL.PVTv2.PRETRAINED_PATH)
#         body.load_state_dict(checkpoint['model'])




# # Dictionary of PVTv2 variants, you should fill in the configuration for each variant based on your needs.
# pvt_variants = {
#     'pvt_v2_b0': {'depths': (2, 2, 2, 2), 'embed_dims': (32, 64, 160, 256), 'num_heads': (1, 2, 5, 8)},
#     'pvt_v2_b1': {'depths': (2, 2, 2, 2), 'embed_dims': (64, 128, 320, 512), 'num_heads': (1, 2, 5, 8)},
#     'pvt_v2_b2': {'depths': (3, 4, 6, 3), 'embed_dims': (64, 128, 320, 512), 'num_heads': (1, 2, 5, 8)},
#     'pvt_v2_b3': {'depths': (3, 4, 18, 3), 'embed_dims': (64, 128, 320, 512), 'num_heads': (1, 2, 5, 8)},
#     'pvt_v2_b4': {'depths': (3, 8, 27, 3), 'embed_dims': (64, 128, 320, 512), 'num_heads': (1, 2, 5, 8)},
#     'pvt_v2_b5': {'depths': (3, 6, 40, 3), 'embed_dims': (64, 128, 320, 512), 'num_heads': (1, 2, 5, 8), 'mlp_ratios': (4, 4, 4, 4)},
#     'pvt_v2_b2_li': {'depths': (3, 4, 6, 3), 'embed_dims': (64, 128, 320, 512), 'num_heads': (1, 2, 5, 8), 'linear': True}
# }


# # Register each PVTv2 backbone variant
# for variant_name, variant_config in pvt_variants.items():
#     # Register the backbone builder function for this variant
#     registry.BACKBONES.register(variant_name)(
#         lambda cfg, variant_config=variant_config: build_pvtv2_fpn_backbone(cfg, variant_config)
#     )



# def build_pvtv2_fpn_backbone(cfg, variant):
#     # Create the PVT model using TIMM with features only
#     body = create_model(
#         variant,
#         pretrained=cfg.MODEL.PVTv2.PRETRAINED,
#         features_only=True,
#         out_indices=(0, 1, 2, 3)  # Modify this if you need different layers
#     )
