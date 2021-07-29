# DCGAN architecture
# Refer to Table 7 of InfoGAN-CR


# ResNet generator and discriminator
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class DCGAN_Generator(nn.Module):
    def __init__(self, dim_z=5, dim_c=5):
        super(DCGAN_Generator, self).__init__()
        self.dim_z = dim_z
        self.dim_c = dim_c

        self.fc = nn.Sequential(
            nn.Linear(self.dim_z+self.dim_c, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 4*4*64),
            nn.BatchNorm1d(4*4*64),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), #h=2h; 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True), #h=2h; 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=True), #h=h*2; 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True), #h=h*2; 64
            nn.Sigmoid()
        )
    
    def forward(self, z, c):
        z = z.view(-1, self.dim_z)
        c = c.view(-1, self.dim_c)
        x = torch.cat((z, c), 1)
        output = self.fc(x)
        output = output.view(-1, 64, 4, 4)
        output = self.deconv(output)
        return output


class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            
            spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 32
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 16
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 8
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 4
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(64*4*4, 128)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(-1, 64*4*4)
        output = self.fc(output)
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



class DCGAN_Discriminator_CR(nn.Module):
    def __init__(self, dim_c=5):
        super(DCGAN_Discriminator_CR, self).__init__()
        self.dim_c = dim_c
        
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1, bias=True), #h=h/2; 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=True), #h=h/2; 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True), #h=h/2; 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), #h=h/2; 4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.dim_c)
        )

    def forward(self, x, xp):
        x = torch.cat((x,xp), dim=1)
        output = self.conv(x)
        output = output.view(-1, 64*4*4)
        output = self.fc(output)
        return output








if __name__ == "__main__":
    netG = DCGAN_Generator(dim_z=5, dim_c=5).cuda()
    netD = DCGAN_Discriminator().cuda()
    net_DHead = DHead().cuda()
    net_QHead = QHead(dim_c=5).cuda()
    net_CR = DCGAN_Discriminator_CR().cuda()
    z = torch.randn(16, 5).cuda()
    c = torch.randn(16, 5).cuda()
    out_G = netG(z,c)
    out_D = netD(out_G)
    out_DH = net_DHead(out_D)
    mu_QH, var_QH = net_QHead(out_D)
    out_CR = net_CR(out_G, out_G)


    print(out_G.size())
    print(out_D.size())
    print(out_DH.size())
    print(mu_QH.size(), var_QH.size())
    print(out_CR.size())

    import numpy as np
    class NormalNLLLoss:
        """
        Calculate the negative log likelihood
        of normal distribution.
        This needs to be minimised.
        Treating Q(cj | x) as a factored Gaussian.
        """
        def __call__(self, x, mu, var):
            
            logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
            nll = -(logli.sum(1).mean())

            return nll
    criterionQ_con = NormalNLLLoss()
    noises = torch.randn(16, 5).cuda()
    loss = criterionQ_con(noises, mu_QH, var_QH)
    print(loss)