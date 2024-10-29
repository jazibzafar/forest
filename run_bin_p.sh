#!/usr/bin/bash

ARCH='vit_small'
CKPT_PATH='/data_hdd/pauline/models/dino_vit_small_random.ckpt'
CKPT_KEY='teacher'
DATA_PATH='/data_hdd/pauline/dataset/'
INPUT_SIZE=320
OUTPUT_DIR="./test/"
MAX_EPOCHS=10


# Check if output_dir exists
if test -d $OUTPUT_DIR; then
  echo "output directory exists"
  else
    echo "output directory does not exist. Creating it."
    mkdir $OUTPUT_DIR
fi

srun -w node4 python light_segmentation.py \
  --arch $ARCH \
  --checkpoint_path $CKPT_PATH \
  --checkpoint_key $CKPT_KEY \
  --data_path $DATA_PATH \
  --freeze_backbone \
  --input_size $INPUT_SIZE \
  --output_dir $OUTPUT_DIR \
  --max_epochs $MAX_EPOCHS