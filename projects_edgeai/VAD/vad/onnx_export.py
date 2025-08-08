import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnxsim import simplify
from mmdet3d.structures.ops import bbox3d2result
from .onnx_network import VAD_export_model

import copy
import onnx


def create_onnx_VAD(model):
    onnxModel = VAD_export_model(model.img_backbone,
                                 model.img_neck,
                                 model.pts_bbox_head,
                                 model.video_test_mode,
                                 model.map_pred2result)
    onnxModel = onnxModel.cpu()
    onnxModel.eval()

    return onnxModel


def export_VAD(onnxModel,
               inputs,
               #img_metas,
               data_samples,
               #gt_bboxes_3d,
               #gt_labels_3d,
               #img=None,
               #ego_his_trajs=None,
               #ego_fut_trajs=None,
               #ego_fut_cmd=None,
               #ego_lcf_feat=None,
               #gt_attr_labels=None,
               **kwargs
            ):

    img = inputs['imgs'].clone().cpu()
    copy_data_samples = copy.deepcopy(data_samples)
    batch_img_metas = [ds.metainfo for ds in copy_data_samples]

    # For temporal info
    if batch_img_metas[0]['scene_token'] != onnxModel.prev_frame_info['scene_token']:
        onnxModel.prev_frame_info['prev_bev'] = None
    onnxModel.prev_frame_info['scene_token'] = batch_img_metas[0]['scene_token']

    # do not use temporal information
    if not onnxModel.video_test_mode:
        onnxModel.prev_frame_info['prev_bev'] = None

    # Get the delta of ego position and angle between two timestamps.
    tmp_pos = copy.deepcopy(batch_img_metas[0]['can_bus'][:3])
    tmp_angle = copy.deepcopy(batch_img_metas[0]['can_bus'][-1])

    if onnxModel.prev_frame_info['prev_bev'] is not None:
        batch_img_metas[0]['can_bus'][:3] -= onnxModel.prev_frame_info['prev_pos']
        batch_img_metas[0]['can_bus'][-1] -= onnxModel.prev_frame_info['prev_angle']
    else:
        batch_img_metas[0]['can_bus'][-1] = 0
        batch_img_metas[0]['can_bus'][:3] = 0

    if onnxModel.prev_frame_info['prev_bev'] is None:
        onnxModel.prev_frame_info['prev_bev'] = torch.zeros(
            [onnxModel.bev_embedding.weight.size(0), 1, onnxModel.bev_embedding.weight.size(1)])

    onnxModel.prepare_data(batch_img_metas)
    reference_points_cam, bev_mask_count, bev_valid_indices, bev_valid_indices_count, shift_xy, can_bus = \
        onnxModel.precompute_bev_info(batch_img_metas)

    rotation_grid = None
    if onnxModel.prev_frame_info['prev_bev'] is not None:
        rotation_grid = onnxModel.compute_rotation_matrix(
            onnxModel.prev_frame_info['prev_bev'], batch_img_metas)

    # Passed the squeezed img
    if img.dim() == 5 and img.size(0) == 1:
        img.squeeze_()
    elif img.dim() == 5 and img.size(0) > 1:
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

    model_input = []
    model_input.append(img)
    model_input.append(shift_xy)
    model_input.append(rotation_grid)
    model_input.append(reference_points_cam)
    model_input.append(bev_mask_count)
    model_input.append(bev_valid_indices)
    model_input.append(bev_valid_indices_count)
    model_input.append(can_bus)
    model_input.append(onnxModel.prev_frame_info['prev_bev'])

    model_name   = 'vad_nus.onnx'
    input_names  = ["inputs", "shift_xy", "rotation_grid", "reference_points_cam",
                    "bev_mask_count", "bev_valid_indices", "bev_valid_indices_count",
                    "can_bus", "prev_bev"]
    output_names = ["bboxes", "scores", "labels",
                    "agent_trajs", "map_bboxes", "map_scores", "map_labels",
                    "map_pts", "ego_fut_preds", "bev_feature"]

    torch.onnx.export(onnxModel,
                      tuple(model_input),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=16,
                      verbose=False)

    onnxModel.prev_frame_info['prev_pos']   = tmp_pos
    onnxModel.prev_frame_info['prev_angle'] = tmp_angle

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    # move model back to gpu
    onnxModel = onnxModel.cuda()

    print("\n!! ONNX model has been exported for VAD!!!\n\n")