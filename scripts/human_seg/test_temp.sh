#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/human_seg \
--name human_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 12 \
--export_folder meshes \
--gpu_ids -1 \