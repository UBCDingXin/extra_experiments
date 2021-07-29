## Path
ROOT_PATH="./RC-49/CVAE"
DATA_PATH="./datasets/RC-49"
EVAL_PATH="./RC-49/evaluation/eval_models"

SEED=2021
NUM_WORKERS=0
MIN_LABEL=0
MAX_LABEL=90
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0


NITERS=50000
BATCH_SIZE=64
LR=3e-4
DIM_Z=256
DIM_C=1

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --niters $NITERS --resume_niters 0 --save_niters_freq 2000 \
    --lr $LR --dim_z $DIM_Z --dim_c $DIM_C --batch_size $BATCH_SIZE \
    --visualize_freq 500 --visualize_fake_images --comp_FID \
    2>&1 | tee output_1.txt
