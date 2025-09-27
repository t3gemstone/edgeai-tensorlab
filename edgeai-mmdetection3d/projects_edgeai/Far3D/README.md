# Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection

> [Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2303.11926)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we propose a long-sequence modeling framework, named StreamPETR, for multi-view 3D object detection. Built upon the sparse query design in the PETR series, we systematically develop an object-centric temporal mechanism. The model is performed in an online manner and the long-term historical information is propagated through object queries frame by frame. Besides, we introduce a motion-aware layer normalization to model the movement of the objects. StreamPETR achieves significant performance improvements only with negligible computation cost, compared to the single-frame baseline. On the standard nuScenes benchmark, it is the first online multi-view method that achieves comparable performance (67.6% NDS & 65.3% AMOTA) with lidar-based methods. The lightweight version realizes 45.0% mAP and 31.7 FPS, outperforming the state-of-the-art method (SOLOFusion) by 2.3% mAP and 1.8x faster FPS. Code has been available at [this URL](https://github.com/exiawsh/StreamPETR).

# Far3D: Expanding the Horizon for Surround-view 3D Object Detection

> [Far3D: Expanding the Horizon for Surround-view 3D Object Detection](https://arxiv.org/abs/2308.09616)

<!-- [ALGORITHM] -->

## Abstract

Recently 3D object detection from surround-view images has made notable advancements with its low deployment cost. However, most works have primarily focused on close perception range while leaving long-range detection less explored. Expanding existing methods directly to cover long distances poses challenges such as heavy computation costs and unstable convergence. To address these limitations, this paper proposes a novel sparse query-based framework, dubbed Far3D. By utilizing high-quality 2D object priors, we generate 3D adaptive queries that complement the 3D global queries. To efficiently capture discriminative features across different views and scales for long-range objects, we introduce a perspective-aware aggregation module. Additionally, we propose a range-modulated 3D denoising approach to address query error propagation and mitigate convergence issues in long-range tasks. Significantly, Far3D demonstrates SoTA performance on the challenging Argoverse 2 dataset, covering a wide range of 150 meters, surpassing several LiDAR-based approaches. Meanwhile, Far3D exhibits superior performance compared to previous methods on the nuScenes dataset. The code is available at [this URL](https://github.com/megvii-research/Far3D). 


## Introduction

We implement and provide the results and checkpoints for StreamPETR and Far3D on the NuScenes dataset.

## Dataset Preperation

### NuScenes

Prepare the nuScenes dataset as per the MMDetection3D documentation [NuScenes Dataset Preperation](../../docs/en/advanced_guides/datasets/nuscenes.md). 

After downloading nuScenes 3D detection dataset and unzipping all zip files, we typically need to organize the useful data information with a `.pkl` file in a specific style. To prepare them with NuScenes dataset for StreamPETR and Far3D, run the following command:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --strpetr
```

The directory structure after processing should be as below.

```
edgeai-mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── lidarseg (optional)
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_gt_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

### PandaSet

StreamPETR and Far3D are not supported yet in edgeai-mmdetection3d. 

## Get Started

Refer the MMDetection3D documentation [Test and Train with Standard Datasets](../../docs/en/user_guides/train_test.md) for general floating point training/evaluation/testing steps for standard datasets. Use the below steps for training and evaluation of BEVDet:

1. cd to installation directory <install_dir>/edgeai-mmdetection3d

2. Do floating-model training using the command 
    "./tools/dist_train.sh <config_file> <num_gpus>"

    For example, to use 2 GPUs use the command

    ```bash
    # StreamPETR with NuScenes
    ./tools/dist_train.sh projects_edgeai/Far3D/configs/nuscenes/streampetr_r50_256x704_bs4_seq_24e.py 2

    # Far3D with NuScenes
    ./tools/dist_train.sh projects_edgeai/Far3D/configs/nuscenes/far3d_vovnet_gridmask_960x640.py 2
    ```

3.  Do evalution using the command 

    "python <config_file> <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    # StreamPETR with NuScenes
    python ./tools/test.py projects_edgeai/StreamPETR/configs/nuscenes/streampetr_r50_256x704_bs4_seq_24e.py ./work_dirs/streampetr_r50_256x704_bs4_seq_24e/epoch_24.pth

    # Far3D with NuScenes
    python ./tools/test.py projects_edgeai/Far3D/configs/nuscenes/far3d_vovnet_gridmask_960x640.py ./work_dirs/far3d_vovnet_gridmask_960x640/epoch_24.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.

## Results

The following results are for StreamPETR and Far#D with NuScenes.


|  Dataset  |                    Model                      | Mem (GB) | Inf time (fps) | mAP    | NDS   |
|:---------:| :-------------------------------------------- | :------: | :------------: | :---:  | :--:  |
| NuScenes  | streampetr_r50_256x704_bs4_seq_24e            |   0.38   |       TBA      | 39.07  | 48.92 |
|           | far3d_vovnet_gridmask_960x640                 |   2.20   |       TBA      | 41.75  | 52.79 |


<!--
## 3D Object Detection Model Zoo

Complexity and Accuracy report of several trained models is available at the [3D Detection Model Zoo](../../docs/det3d_modelzoo.md) 


## Quantization
This tutorial explains more about quantization and how to do [Quantization Aware Training (QAT)](../../docs/det3d_quantization.md) of detection models.
-->

## ONNX & Prototxt Export

Export of ONNX model (.onnx) is supported by setting the field `save_onnx_model`=True in `model` in a config file. For example,

```bash
model = dict(
    type='StreamPETR',
    save_onnx_model=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        ...
```

## References

[1] Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection, S. Wang, Y. Liu, T. Wang, X. Zhang, https://arxiv.org/abs/2303.11926

[2] Far3D: Expanding the Horizon for Surround-view 3D Object Detection, X. Jiang, S. Li, Y. Liu, S. Wang, F. Jia, T. Wang, L. Han, X. Zhang, https://arxiv.org/abs/2308.09616

