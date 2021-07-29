## ref to https://github.com/yt4766269/Attribute2Image/blob/master/condVAE.py and https://github.com/xcyan/eccv16_attr2img/blob/master/scripts/arch_condVAE.lua

import torch
import torch.nn as nn
import torch.nn.functional as function
import copy

class CVAE_encoder(nn.Module):
    def __init__(self, zdim:int, ydim:int):

        super(CVAE_encoder, self).__init__()
        self.x_encoder_conv = nn.Sequential(
            # -- 64 x 64 --> 32 x 32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2), #h=h
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #h=h/2
            # 32x32 -> 16x16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2), #h=h
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #h=h/2
            # 16x16 -> 8x8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  #h=h
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #h=h/2
            # 8x8 -> 4x4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1), #h=h
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # 4x4 -> 1x1
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.x_encoder_full = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(in_features=ydim, out_features=zdim),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(in_features=1024+zdim, out_features=zdim)
        self.linear2 = nn.Linear(in_features=1024+zdim, out_features=zdim)
    def forward(self, x, y):
        y = y.view(-1,1)
        x = self.x_encoder_conv(x)
        x = x.view(-1, 1024)
        
        x = self.x_encoder_full(x)
        y = self.y_encoder(y)
        cat = torch.cat([x, y],dim=1)
        mean = self.linear1(cat)
        var = self.linear2(cat) + 1e-6
        return mean, var
    

class CVAE_decoder(nn.Module):
    def __init__(self, zdim:int, ydim:int):
        super(CVAE_decoder, self).__init__()
        self.y_decode = nn.Sequential(
            nn.Linear(in_features=ydim, out_features=zdim*2),
            nn.ReLU()
        )
        self.linearMix = nn.Linear(in_features=zdim*3, out_features=256)

        self.linear1 = nn.Linear(in_features=256, out_features=256*8*8)
        self.conv_decoder = nn.Sequential(
            # 8x8 -> 8x8
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # 8x8->16x16
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # 16 x 16 --> 32 x 32
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # 32 x 32 --> 64 x 64
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # 64 x 64 --> 64 x 64
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=3),
            # nn.ReLU()
            nn.Sigmoid()
        )
    def forward(self, z, y):
        y = y.view(-1,1)
        y = self.y_decode(y)
        z = torch.cat((y,z),dim=1)
        z = self.linearMix(z)
        z = function.relu(z)
        z = self.linear1(z)
        z = z.view(-1, 256,8,8)
        x = self.conv_decoder(z)
        return x



if __name__ == "__main__":
    encoder = CVAE_encoder(zdim=256, ydim=1).cuda()
    decoder = CVAE_decoder(zdim=256, ydim=1).cuda()
    z = torch.randn(16, 256).cuda()
    c = torch.randn(16, 1).cuda()
    x = decoder(z,c)
    print(x.size())
    mu,var = encoder(x,c)
    print(mu.size())
    print(var.size())