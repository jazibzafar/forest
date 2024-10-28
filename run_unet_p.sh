#!/usr/bin/bash

ARCH='resnet50'
CKPT_PATH='/data_hdd/jazibmodels/dino_vit-s_32_500k_randonly/epoch=6-step=500000.ckpt'
CKPT_KEY="teacher"
DATA_PATH='/data_hdd/pauline/dataset/'
NUM_CLASSES=4
OUTPUT_DIR="./test_unet/"
MAX_EPOCHS=10

# Check if output_dir exists
if test -d $OUTPUT_DIR; then
  echo "output directory exists"
  else
    echo "output directory does not exist. Creating it."
    mkdir $OUTPUT_DIR
fi

srun -w node4 python light_unet.py \
  --arch $ARCH \
  --checkpoint_path $CKPT_PATH \
  --checkpoint_key $CKPT_KEY \
  --data_path $DATA_PATH \
  --num_classes $NUM_CLASSES \
  --output_dir $OUTPUT_DIR \
  --max_epochs $MAX_EPOCHS