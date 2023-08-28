# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

import torch
from torch import nn
import time

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_pt_lab_ten, xyz_ten, batch_size, configs):
        coords, features_3d, voxel_labels = self.cylinder_3d_generator(train_pt_fea_ten, train_pt_lab_ten, xyz_ten, 
                                                                       batch_size, configs)
        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return spatial_features, voxel_labels
