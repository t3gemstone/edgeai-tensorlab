# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs,  to_tensor
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D

from mmengine.structures import InstanceData

import mmcv
import warnings

@TRANSFORMS.register_module()
class CustomMultiScaleFlipAug3D(MultiScaleFlipAug3D):    
    def transform(self, results):
        results = super().transform(results)
        if isinstance(results, list):
            results = results[0]

        return results

