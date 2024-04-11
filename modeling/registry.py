# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fcos.core.utils.registry import Registry

BACKBONES = Registry()
RPN_HEADS = Registry()
ROI_BOX_FEATURE_EXTRACTORS = Registry()
ROI_BOX_PREDICTOR = Registry()
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
ROI_KEYPOINT_PREDICTOR = Registry()
ROI_MASK_FEATURE_EXTRACTORS = Registry()
ROI_MASK_PREDICTOR = Registry()

ALIGNMENT_MODULE = Registry()
ATTENTION_MODULE = Registry()
FUSION_MODULE = Registry() 