# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import _import_file as import_file

from . import datasets as D
from . import samplers


def build_dataset(dataset_list, transforms, dataset_catalog, cfg=None, is_train=True, mode='train', rng=None):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        args = data["args"]
        if mode == 'support' or (mode == 'finetune' and is_train) :
            data["factory"] = mode.capitalize() + data["factory"]
            args["cfg"] = cfg
            args["rng"] = rng
            args["remove_images_without_annotations"] = True
        factory = getattr(D, data["factory"])

        # for COCODataset, we want to remove images without annotations
        # during training
        if "COCODataset" in data["factory"]:
            args["remove_images_without_annotations"] = is_train
        if "PascalVOCDataset" in data["factory"]:
            args["use_difficult"] = not is_train
        args["transforms"] = transforms

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0, is_fewshot=False, is_support=False):
    if is_fewshot:
        return BatchSampler(sampler, images_per_batch, drop_last=False)
    elif aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        if is_support:
            batch_sampler = samplers.SupportGroupedBatchSampler(sampler, group_ids, images_per_batch, drop_uneven=False)
        else:
            batch_sampler = samplers.GroupedBatchSampler(sampler, group_ids, images_per_batch, drop_uneven=False)
    else:
        batch_sampler = BatchSampler(sampler, images_per_batch, drop_last=False)
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler
