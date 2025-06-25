#!/usr/bin/env bash

# PYTHONPATH must start with a : to be able to load local modules
# but this can cause a confusion the installed torchvision and the local torchvision
# export PYTHONPATH=:$PYTHONPATH

# Date/time in YYYYMMDD-HHmmSS format
DATE_TIME=`date +'%Y%m%d-%H%M%S'`

#=========================================================================================
# these lite models will be available only if --model-surgery <argument> argument is set
# --model-surgery 1: legacy module based surgery
# --model-surgery 2: advanced model surgery with torch.fx (to be released)
model=ssdlite320_mobilenet_v3_large

#=========================================================================================
output_dir="./data/checkpoints/torchvision/coco_detection_${model}"

#=========================================================================================
torchrun --nproc_per_node 4 ./references/detection/train.py --data-path ./data/datasets/coco --model ${model} \
--epochs=1 --aspect-ratio-group-factor 3 --lr-scheduler cosineannealinglr --lr 0.15 \
--batch-size 16 --weight-decay 0.00004 --data-augmentation ssdlite \
--model-surgery 1 --output-dir=${output_dir}
