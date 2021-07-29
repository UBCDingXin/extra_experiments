'''
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer: https://github.com/pfnet-research/sngan_projection
'''

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm


channels = 3
GEN_SIZE=64
DISC_SIZE=64



class ResBlockGenerator(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=True) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        out = self.model(x) + self.bypass(x)
        return out


class SNGAN_Generator(nn.Module):
    def __init__(self, dim_z=5, dim_c=5):
        super(SNGAN_Generator, self).__init__()
        self.dim_z = dim_z
        self.dim_c = dim_c

        self.dense = nn.Linear(self.dim_z+self.dim_c, 4 * 4 * GEN_SIZE*8, bias=True)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*8) #4--->8
        self.genblock1 = ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4) #8--->16
        self.genblock2 = ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2) #16--->32
        self.genblock3 = ResBlockGenerator(GEN_SIZE*2, GEN_SIZE) #32--->64

        self.final = nn.Sequential(
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, c):
        z = z.view(-1, self.dim_z)
        c = c.view(-1, self.dim_c)
        x = torch.cat((z, c), 1)
        out = self.dense(x)
        out = out.view(-1, GEN_SIZE*8, 4, 4)

        out = self.genblock0(out)
        out = self.genblock1(out)
        out = self.genblock2(out)
        out = self.genblock3(out)
        out = self.final(out)

        return out



class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class SNGAN_Discriminator(nn.Module):
    def __init__(self):
        super(SNGAN_Discriminator, self).__init__()

        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2), #64--->32
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE*2, stride=2), #32--->16
            ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2), #16--->8
        )
        self.discblock2 = ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2) #8--->4
        self.discblock3 = nn.Sequential(
            ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*8, stride=1), #4--->4;
            nn.ReLU(),
        )

        self.linear = nn.Linear(DISC_SIZE*8*4*4, 128, bias=True)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
        self.linear = spectral_norm(self.linear)

    def forward(self, x):
        output = self.discblock1(x)
        output = self.discblock2(output)
        output = self.discblock3(output)

        output = output.view(-1, DISC_SIZE*8*4*4)
        output = F.leaky_relu(self.linear(output), negative_slope=0.2, inplace=True)

        return output



## DHead and QHead are based on https://github.com/Natsu6767/InfoGAN-PyTorch/blob/4586919f2821b9b2e4aeff8a07c5003a5905c7f9/models/mnist_model.py#L52
## and Table 7 of InfoGAN-CR
class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_disc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc_disc(x)

class QHead(nn.Module):
    def __init__(self, dim_c):
        super().__init__()

        self.fc_qnet = nn.Sequential(
            spectral_norm(nn.Linear(128, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Linear(128, dim_c)),
        )

        self.fc_mu = nn.Linear(128, dim_c)
        self.fc_var = nn.Linear(128, dim_c)

    def forward(self, x):
        out = self.fc_qnet(x)
        return self.fc_mu(out), torch.exp(self.fc_var(out))




if __name__ == "__main__":
    netG = SNGAN_Generator(dim_z=5, dim_c=5).cuda()
    netD = SNGAN_Discriminator().cuda()
    net_DHead = DHead().cuda()
    net_QHead = QHead(dim_c=5).cuda()
    z = torch.randn(16, 5).cuda()
    c = torch.randn(16, 5).cuda()
    out_G = netG(z,c)
    out_D = netD(out_G)
    out_DH = net_DHead(out_D)
    mu_QH, var_QH = net_QHead(out_D)
    print(out_G.size())
    print(out_D.size())
    print(out_DH.size())
    print(mu_QH.size(), var_QH.size())

    import numpy as np
    class NormalNLLLoss:
        def __call__(self, x, mu, var):
            
            logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
            nll = -(logli.sum(1).mean())

            return nll
    criterionQ_con = NormalNLLLoss()
    noises = torch.randn(16, 5).cuda()
    loss = criterionQ_con(noises, mu_QH, var_QH)
    print(loss)