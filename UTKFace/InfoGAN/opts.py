import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='G:/OneDrive/Working_directory/CcGAN_TPAMI/UTKFace/UTKFace_64x64/InfoGAN')
    parser.add_argument('--data_path', type=str, default='G:/OneDrive/Working_directory/CcGAN_TPAMI/datasets/UTKFace')
    parser.add_argument('--eval_ckpt_path', type=str, default='G:/OneDrive/Working_directory/CcGAN_TPAMI/UTKFace/UTKFace_64x64/evaluation/eval_models')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 2021)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Dataset '''
    parser.add_argument('--min_label', type=int, default=1, metavar='N')
    parser.add_argument('--max_label', type=int, default=60, metavar='N')
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N', choices=[64, 128, 256])
    parser.add_argument('--max_num_img_per_label', type=int, default=2000, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=200, metavar='N')
    parser.add_argument('--show_real_imgs', action='store_true', default=False)
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)

    ''' GAN settings '''
    parser.add_argument('--GAN_arch', type=str, default='DCGAN', choices=['DCGAN', 'SNGAN'])
    parser.add_argument('--niters_gan', type=int, default=100000, help='number of iterations')
    parser.add_argument('--resume_niters_gan', type=int, default=0)
    parser.add_argument('--save_niters_freq', type=int, default=2000, help='frequency of saving checkpoints')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='learning rate for discriminator')
    parser.add_argument('--dim_z', type=int, default=5, help='dim of noise')
    parser.add_argument('--dim_c', type=int, default=5, help='dim of latent continuous codes')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lambda_info', type=float, default=0.2, help='lambda for the info loss')
    parser.add_argument('--visualize_freq', type=int, default=2000, help='frequency of visualization')

    # evaluation setting
    parser.add_argument('--label_loc', type=int, default=0)
    parser.add_argument('--reverse_label', action='store_true', default=False)
    parser.add_argument('--samp_batch_size', type=int, default=1000)
    parser.add_argument('--nfake_per_label', type=int, default=1000)
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=int, default=5)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)

    args = parser.parse_args()
    return args