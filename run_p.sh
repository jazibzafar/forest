#!/usr/bin/bash

OUTPUT_DIR="./test/"

# Check if output_dir exists
if test -d $OUTPUT_DIR; then
  echo "output directory exists"
  else
    echo "output directory does not exist. Creating it."
    mkdir $OUTPUT_DIR
fi

srun -w node4 python light_segmentation.py