## Path
ROOT_PATH="./UTKFace//InfoGAN"
DATA_PATH="./datasets/UTKFace"
EVAL_PATH="./UTKFace/evaluation/eval_models"

## setting
SEED=2021
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=2000
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

## GAN setting
NITERS=50000
BATCH_SIZE=64
LR_G=2e-4
LR_D=2e-4
DIM_Z=5
DIM_C=5
LAMBDA_INFO=0.2
GAN_ARCH="DCGAN"

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
    --lr_g $LR_G --lr_d $LR_D --dim_z $DIM_Z --dim_c $DIM_C --batch_size $BATCH_SIZE --lambda_info $LAMBDA_INFO \
    --visualize_freq 2000 --visualize_fake_images \
    2>&1 | tee output_1.txt
