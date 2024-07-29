#!/usr/bin/bash

#python sequential_classification.py --yaml_file 'dino_seq.yaml'
#python single_classification.py --yaml_file 'dino_args.yaml'
#python single_segmentation.py --yaml_file 'dino_segmentation.yaml'
python bench_unet.py --yaml_file 'dino_segmentation.yaml'