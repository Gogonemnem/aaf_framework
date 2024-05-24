from timm import create_model

from fcos.core.modeling import registry
from .fpn import FPN

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

def build_pvtv2(cfg, variant):
    # Create the model using TIMM
    model = create_model(
        variant,
        pretrained=cfg.PRETRAINED_BACKBONE,
        features_only=True,
        out_indices=(0, 1, 2, 3)
    )

    # Dynamically add the _freeze_backbone method to the body object to conform to fcos package
    setattr(model, '_freeze_backbone', _freeze_backbone.__get__(model, type(model)))
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

# Register all PVTv2 backbones with different FPN configurations
for uppercase_variant, lowercase_variant in variant_name_mapping.items():
    registry.BACKBONES.register(f"{uppercase_variant}")(
        lambda cfg: build_pvtv2(cfg, lowercase_variant)
    )

    registry.BACKBONES.register(f"{uppercase_variant}-FPN")(
        lambda cfg: FPN(cfg, build_pvtv2(cfg, lowercase_variant), fpn_type="original").build()
    )

    registry.BACKBONES.register(f"{uppercase_variant}-FPN-RETINANET")(
        lambda cfg: FPN(cfg, build_pvtv2(cfg, lowercase_variant), fpn_type="retinanet").build()
    )

    registry.BACKBONES.register(f"{uppercase_variant}-FPN-HYBRID")(
        lambda cfg: FPN(cfg, build_pvtv2(cfg, lowercase_variant), fpn_type="hybrid").build()
    )