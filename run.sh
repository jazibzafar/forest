#!/usr/bin/bash
#SBATCH -w node5 --gpus 1

echo "Pretraining DINO"

ARCH='resnet50'
BATCH_SIZE=32
MAX_STEPS=1500000
WARMUP_STEPS=50000
FREEZE_LAST_LAYER=15000
LR=1e-3
WEIGHT_DECAY_INIT=1e-4
WEIGHT_DECAY_END=1e-4
MIN_LR=1e-6  # redundant as min_lr is hardcoded where needed.
GLOBAL_CROP_SIZE=192
LOCAL_CROP_SIZE=96
DATA_PATH='/data_local_ssd/random_25k_tars/random_25k_{0000..0099}.tar'
#DATA_PATH='/data_local_ssd/nrw_dop10_tars/nrw_dop10-{0000..0099}.tar'
#DATA_PATH='/data_local_ssd/nrw_dop10_tars/nrw_dop10-0000.tar'
#RESUME='./exp_240304/last.ckpt'
OUTPUT_DIR='/data_hdd/jazibmodels/dino_resnet50_32_1.5m_randonly/'
NUM_WORKERS=8


# Check if output_dir exists
if test -d $OUTPUT_DIR; then
  echo "output directory exists"
  else
    echo "output directory does not exist. Creating it."
    mkdir $OUTPUT_DIR
fi

# Run the training code.
srun -w node5 python light_pretrain_dino.py \
  --arch $ARCH \
  --batch_size_per_gpu $BATCH_SIZE \
  --max_steps $MAX_STEPS \
  --freeze_last_layer $FREEZE_LAST_LAYER \
  --lr $LR \
  --warmup_steps $WARMUP_STEPS \
  --min_lr $MIN_LR \
  --weight_decay_init $WEIGHT_DECAY_INIT \
  --weight_decay_end $WEIGHT_DECAY_END \
  --global_crop_size $GLOBAL_CROP_SIZE \
  --local_crop_size $LOCAL_CROP_SIZE \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_workers $NUM_WORKERS
  #  --resume $RESUME \
