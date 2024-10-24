#!/usr/bin/bash

#OUTPUT_DIR="./test/"
##EXP_NAME="second/"
#
#
## Check if output_dir exists
#if test -d $OUTPUT_DIR; then
#  echo "output directory exists"
#  else
#    echo "output directory does not exist. Creating it."
#    mkdir $OUTPUT_DIR
#fi

srun -w node4 python light_segmentation.py