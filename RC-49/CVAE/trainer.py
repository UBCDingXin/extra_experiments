import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import IMGs_dataset, SimpleProgressBar
from opts import parse_opts

''' Settings '''
args = parse_opts()

# some parameters in opts
niters = args.niters
resume_niters = args.resume_niters
dim_z = args.dim_z
dim_c = args.dim_c
lr = args.lr
save_niters_freq = args.save_niters_freq
visualize_freq = args.visualize_freq
batch_size = args.batch_size
num_channels = args.num_channels
img_size = args.img_size
num_workers = args.num_workers
max_label = args.max_label


# def loss_fn(recon_x, x, mean, log_var):
#     BCE = torch.nn.functional.binary_cross_entropy(
#         recon_x.view(-1, num_channels*img_size*img_size), x.view(-1, num_channels*img_size*img_size), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

#     return (BCE + KLD) / x.size(0)

BCE_loss = nn.BCELoss(reduction = "sum")
def loss_fn(X_hat, X, mean, logvar):
    reconstruction_loss = BCE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence


def train_cvae(images, labels, net_encoder, net_decoder, save_images_folder, save_models_folder = None):

    ## init models
    net_encoder = net_encoder.cuda()
    net_decoder = net_decoder.cuda()

    ## optimizer
    optimizer = torch.optim.Adam([{'params': net_encoder.parameters()}, {'params': net_decoder.parameters()}], lr=lr, betas=(0.9, 0.999), weight_decay=0.004)

    ## datasets
    trainset = IMGs_dataset(images, labels=labels, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ## load pre-trained folders
    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/CVAE_checkpoint_intrain/CVAE_checkpoint_niters_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        net_encoder.load_state_dict(checkpoint['net_encoder_state_dict'])
        net_decoder.load_state_dict(checkpoint['net_decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    #################
    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    n_row=10; n_col = n_row
    z_fixed = torch.randn(n_row*n_col, dim_z, dtype=torch.float).cuda()
    start_label = np.quantile(labels, 0.05)
    end_label = np.quantile(labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row*n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i*n_col+j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).cuda()


    ## training loop
    batch_idx = 0
    dataloader_iter = iter(train_dataloader)

    train_loss_list = []

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):
        
        net_encoder.train()
        net_decoder.train()

        if batch_idx+1 == len(train_dataloader):
            dataloader_iter = iter(train_dataloader)
            batch_idx = 0

        # training images
        batch_train_images, batch_train_labels = dataloader_iter.next()
        assert batch_size == batch_train_images.shape[0] and batch_size == batch_train_labels.shape[0]
        batch_train_images = batch_train_images.type(torch.float).cuda()
        batch_train_labels = batch_train_labels.type(torch.float).cuda()

        ## encode
        mean, var = net_encoder(batch_train_images, batch_train_labels)

        ## reprameterization
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        ## decode
        batch_fake_images  = net_decoder(z, batch_train_labels)

        ## cvae loss
        loss = loss_fn(batch_fake_images, batch_train_images, mean, var)

        ## backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        '''

        Print, save ckpts

        '''

        train_loss_list.append(loss.item())

        batch_idx+=1
        if (niter+1)%20 == 0:
            print ("CVAE: [Iter %d/%d] [loss: %.4f] [Time: %.4f]" % (niter+1, niters, loss.item(), timeit.default_timer()-start_time))

        if (niter+1) % visualize_freq == 0:
            net_decoder.eval()
            with torch.no_grad():
                gen_imgs = net_decoder(z_fixed, y_fixed)
                gen_imgs = gen_imgs.detach().cpu()
                save_image(gen_imgs.data, save_images_folder + '/{}.png'.format(niter+1), nrow=n_row, normalize=True)

    #end for niter
    return net_encoder, net_decoder, train_loss_list


def sample_cvae_given_labels(net_decoder, given_labels, batch_size = 500):
    '''
    net_decoder: pretrained decoder network
    given_labels: float. unnormalized labels. we need to convert them to values in [0,1]. 
    '''
    nfake = len(given_labels)
    if batch_size>nfake:
        batch_size=nfake

    ## normalize regression labels to [-1,1] to fit into InfoGAN's framework
    labels = given_labels/max_label

    net_decoder = net_decoder.cuda()
    net_decoder.eval()

    ## concat to avoid out of index errors
    labels = np.concatenate((labels, labels[0:batch_size]), axis=0)

    fake_images = []
    with torch.no_grad():
        pb = SimpleProgressBar()
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, dim_z, dtype=torch.float).cuda()
            c = torch.from_numpy(labels[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            batch_fake_images = net_decoder(z, c)
            fake_images.append(batch_fake_images.detach().cpu().numpy())
            tmp += batch_size
            pb.update(min(float(tmp)/nfake, 1)*100)

    fake_images = np.concatenate(fake_images, axis=0)
    #remove extra images
    fake_images = fake_images[0:nfake]

    #denomarlized fake images
    if fake_images.max()<=1.0:
        # fake_images = ((fake_images*0.5+0.5)*255.0).astype(np.uint8)
        fake_images = (fake_images*255.0).astype(np.uint8)

    return fake_images, given_labels