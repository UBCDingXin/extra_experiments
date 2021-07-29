## Path
ROOT_PATH="./RC-49/InfoGAN-CR"
DATA_PATH="./datasets/RC-49"
EVAL_PATH="./RC-49/evaluation/eval_models"

## setting
SEED=2021
NUM_WORKERS=3
MIN_LABEL=0
MAX_LABEL=90
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

## GAN setting
NITERS=200000
BATCH_SIZE=64
LR_G=1e-4
LR_D=2e-4
DIM_Z=5
DIM_C=3
LAMBDA_INFO=0.2
ALPHA_CR=3.0
GAN_ARCH="DCGAN"

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
    --lr_g $LR_G --lr_d $LR_D --dim_z $DIM_Z --dim_c $DIM_C --batch_size $BATCH_SIZE --lambda_info $LAMBDA_INFO \
    --alpha_cr $ALPHA_CR --cr_gap 1.9 --cr_gap_tran_niter 100000 \
    --visualize_freq 500 \
    2>&1 | tee output_1.txt
