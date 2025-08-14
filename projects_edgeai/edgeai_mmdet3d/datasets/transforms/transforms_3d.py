# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.test_time_aug import MultiScaleFlipAug3D


@TRANSFORMS.register_module()
class CustomMultiScaleFlipAug3D(MultiScaleFlipAug3D):
    def transform(self, results):
        results = super().transform(results)
        if isinstance(results, list):
            results = results[0]

        return results

