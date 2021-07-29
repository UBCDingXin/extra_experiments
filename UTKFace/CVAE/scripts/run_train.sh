## Path
ROOT_PATH="./UTKFace/CVAE"
DATA_PATH="./datasets/UTKFace"
EVAL_PATH="./UTKFace/evaluation/eval_models"

SEED=2021
NUM_WORKERS=3
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=2000
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200


NITERS=100000
BATCH_SIZE=128
LR=3e-4
DIM_Z=256
DIM_C=1

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --niters $NITERS --resume_niters 0 --save_niters_freq 2000 \
    --lr $LR --dim_z $DIM_Z --dim_c $DIM_C --batch_size $BATCH_SIZE \
    --visualize_freq 1000 --visualize_fake_images --comp_FID \
    2>&1 | tee output_1.txt
