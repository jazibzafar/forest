#!/usr/bin/bash


ARCH="vit_small"
CKPT_PATH="/data_hdd/jazibmodels/dino_vit-s_32_500k_randonly/epoch\=6-step\=500000.ckpt"
DATA_PATH="/data_hdd/jazibsdata/gartow_single_class_sliced/8020/"
INPUT_SIZE=224
OUTPUT_DIR="./temp/"

# Check if output_dir exists
if test -d $OUTPUT_DIR; then
  echo "output directory exists"
  else
    echo "output directory does not exist. Creating it."
    mkdir $OUTPUT_DIR
fi

python light_segmentation.py \
  --arch $ARCH \
  --checkpoint_path $CKPT_PATH \
  --data_path $DATA_PATH \
  --input_size $INPUT_SIZE \
  --output_dir $OUTPUT_DIR \
  --freeze_backbone