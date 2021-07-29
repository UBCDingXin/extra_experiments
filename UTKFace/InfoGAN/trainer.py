import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import IMGs_dataset, NormalNLLLoss, weights_init_normal, SimpleProgressBar
from opts import parse_opts

''' Settings '''
args = parse_opts()

# some parameters in opts
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_z = args.dim_z
dim_c = args.dim_c
lambda_info = args.lambda_info
lr_g = args.lr_g
lr_d = args.lr_d
save_niters_freq = args.save_niters_freq
visualize_freq = args.visualize_freq
batch_size = args.batch_size
num_channels = args.num_channels
img_size = args.img_size
num_workers = args.num_workers
max_label = args.max_label
gan_arch = args.GAN_arch


def train_infogan(images, netG, netD, netDH, netQH, save_images_folder, save_models_folder):

    ## init models
    netG = netG.cuda()
    netG.apply(weights_init_normal)
    netD = netD.cuda()
    netD.apply(weights_init_normal)
    netDH = netDH.cuda()
    netDH.apply(weights_init_normal)
    netQH = netQH.cuda()
    netQH.apply(weights_init_normal)

    ## loss functions
    criterionD = nn.BCELoss() #for discriminator
    criterionQ = NormalNLLLoss() # Loss for continuous latent code.


    ## optimizers
    optimizerD = torch.optim.Adam([{'params': netD.parameters()}, {'params': netDH.parameters()}], lr=lr_d, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam([{'params': netG.parameters()}, {'params': netQH.parameters()}], lr=lr_g, betas=(0.5, 0.999))


    ## datasets
    trainset = IMGs_dataset(images, labels=None, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    

    ## load pre-trained folders
    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/InfoGAN_checkpoint_intrain/InfoGAN_checkpoint_niters_{}.pth".format( resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        netDH.load_state_dict(checkpoint['netDH_state_dict'])
        netQH.load_state_dict(checkpoint['netQH_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    ## folder to store fake images generated during training
    ## for each latent code, we vary it from -1 to 1 but randomly set other latent codes and noise
    n_row = 10 #10 distinct labels
    n_col = 10 #for each distinct label generate n_col images
    save_image_folders_list = []
    z_fixed_list = []
    c_fixed_list = []
    for i in range(dim_c):
        save_image_folder_i = save_images_folder + '/latent_code_{}'.format(i)
        save_image_folders_list.append( save_image_folder_i)
        os.makedirs( save_image_folder_i, exist_ok=True)
        
        z_fixed_i = torch.randn(n_row*n_col, dim_z, dtype=torch.float).cuda()
        z_fixed_list.append(z_fixed_i)

        code_range_i = np.linspace(-0.95,0.95,n_row)
        c_vary_i = np.zeros(n_row*n_col)
        for tmp_i in range(n_row):
            curr_code = code_range_i[tmp_i]
            for tmp_j in range(n_col):
                c_vary_i[tmp_i*n_col+tmp_j] = curr_code
        c_vary_i = torch.from_numpy(c_vary_i).type(torch.float).view(-1,1)

        c_fixed_i = []
        for tmp_j in range(dim_c):
            if tmp_j==i:
                c_fixed_i.append(c_vary_i)
            else:
                c_fixed_i.append(((-1 - 1) * torch.rand(n_row*n_col, 1) + 1))
        c_fixed_i = torch.cat(c_fixed_i, dim=1)
        c_fixed_list.append(c_fixed_i.cuda())
        # print(c_fixed_i)

    

    ## training loop
    batch_idx = 0
    dataloader_iter = iter(train_dataloader)

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        if batch_idx+1 == len(train_dataloader):
            dataloader_iter = iter(train_dataloader)
            batch_idx = 0

        # training images
        batch_train_images = dataloader_iter.next()
        assert batch_size == batch_train_images.shape[0]
        batch_train_images = batch_train_images.type(torch.float).cuda()

        # Adversarial ground truths
        GAN_real = torch.ones(batch_size,1).cuda()
        GAN_fake = torch.zeros(batch_size,1).cuda()



        '''

        Train Generator: maximize log(D(G(z)))

        '''
        netG.train()

        # Sample noise as generator input
        z = torch.randn(batch_size, dim_z, dtype=torch.float).cuda()

        # Sample latent codes from U(-1,1)
        c = ((-1 - 1) * torch.rand(batch_size, dim_c) + 1).cuda()

        #generate fake images
        batch_fake_images = netG(z, c)

        # Loss measures generator's ability to fool the discriminator
        dis_out = netD(batch_fake_images)

        # for GAN loss
        DH_out = netDH(dis_out)

        # for info loss
        q_mu, q_var = netQH(dis_out)

        #generator try to let disc believe gen_imgs are real
        g_loss = criterionD(DH_out, GAN_real) + lambda_info * criterionQ(c, q_mu, q_var)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()




        '''

        Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

        '''

        # Measure discriminator's ability to classify real from generated samples
        prob_real = netDH(netD(batch_train_images))
        prob_fake = netDH(netD(batch_fake_images.detach()))
        real_loss = criterionD(prob_real, GAN_real)
        fake_loss = criterionD(prob_fake, GAN_fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()





        '''

        Print, save ckpts

        '''


        batch_idx+=1

        if (niter+1)%20 == 0:
            print ("InfoGAN: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [D prob real:%.4f] [D prob fake:%.4f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), prob_real.mean().item(),prob_fake.mean().item(), timeit.default_timer()-start_time))


        if (niter+1) % visualize_freq == 0:
            netG.eval()
            # with torch.no_grad():
            #     gen_imgs = netG(z_fixed, c_fixed)
            #     gen_imgs = gen_imgs.detach()
            # save_image(gen_imgs.data, save_images_folder +'/{}.png'.format(niter+1), n_row=n_row, normalize=True)

            for tmp_i in range(dim_c):
                z_fixed_i = z_fixed_list[tmp_i]
                c_fixed_i = c_fixed_list[tmp_i]
                save_image_folder_i = save_image_folders_list[tmp_i]
                with torch.no_grad():
                    gen_imgs = netG(z_fixed_i, c_fixed_i)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_image_folder_i +'/{}.png'.format(niter+1), nrow=n_row, normalize=True)



        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/InfoGAN_checkpoint_intrain/InfoGAN_checkpoint_niters_{}.pth".format(niter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netDH_state_dict': netDH.state_dict(),
                    'netQH_state_dict': netQH.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

    #end for niter


    return netG, netD, netDH, netQH





def sample_infogan_given_labels(netG, given_labels, label_loc, reverse_label=False, batch_size = 500):
    '''
    netG: pretrained generator network
    given_labels: float. unnormalized labels. we need to convert them to values in [-1,1]. 
    label_loc: int. location of this label in the continous latent codes of InfoGAN. Note that we let codes randomly vary.
    reverse_label: bool. because in infoGAN, before training, we don't know whether c=-1 equals to the smallest label or the largest label.
    '''

    ## num of fake images will be generated
    nfake = len(given_labels)

    ## normalize regression labels to [-1,1] to fit into InfoGAN's framework
    labels = (given_labels/max_label-0.5)/0.5 

    ## because in infoGAN, before training, we don't know whether c=-1 equals to the smallest label or vice versa.
    if reverse_label:
        labels = -labels

    assert 0<=label_loc<dim_c
    c_all = []
    for i in range(dim_c):
        if i==label_loc:
            c_all.append(labels.reshape(-1,1))
        else:
            c_i = np.random.uniform(-1,1,(nfake,1))
            c_all.append(c_i.reshape(-1,1))
    c_all = np.concatenate(c_all, axis=1)
    assert c_all.shape[0]==nfake and c_all.shape[1]==dim_c

    ## generate images
    if batch_size>nfake:
        batch_size = nfake

    netG=netG.cuda()
    netG.eval()

    ## concat to avoid out of index errors
    c_all = np.concatenate((c_all, c_all[0:batch_size]), axis=0)

    fake_images = []

    with torch.no_grad():
        pb = SimpleProgressBar()
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, dim_z, dtype=torch.float).cuda()
            c = torch.from_numpy(c_all[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            batch_fake_images = netG(z, c)
            fake_images.append(batch_fake_images.detach().cpu().numpy())
            tmp += batch_size
            pb.update(min(float(tmp)/nfake, 1)*100)

    fake_images = np.concatenate(fake_images, axis=0)
    #remove extra images
    fake_images = fake_images[0:nfake]

    #denomarlized fake images
    if fake_images.max()<=1.0:
        fake_images = (fake_images*255.0).astype(np.uint8)
        
    return fake_images, given_labels