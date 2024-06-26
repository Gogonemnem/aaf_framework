# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from detectron2.utils.registry import Registry

BACKBONES = Registry("BACKBONE")
RPN_HEADS = Registry("RPN_HEAD")
ROI_BOX_FEATURE_EXTRACTORS = Registry("ROI_BOX_FEATURE_EXTRACTOR")
ROI_BOX_PREDICTOR = Registry("ROI_BOX_PREDICTOR")
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry("ROI_KEYPOINT_FEATURE_EXTRACTOR")
ROI_KEYPOINT_PREDICTOR = Registry("ROI_KEYPOINT_PREDICTOR")
ROI_MASK_FEATURE_EXTRACTORS = Registry("ROI_MASK_FEATURE_EXTRACTOR")
ROI_MASK_PREDICTOR = Registry("ROI_MASK_PREDICTOR")

ALIGNMENT_MODULE = Registry("ALIGNMENT_MODULE")
ATTENTION_MODULE = Registry("ATTENTION_MODULE")
FUSION_MODULE = Registry("FUSION_MODULE")