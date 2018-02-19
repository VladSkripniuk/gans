import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SNConv2d import SNConv2d


# https://github.com/sunshineatnoon/Paper-Implementations/tree/master/dcgan

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class mnistnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False): # 256 ok
        super(mnistnet_G,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*4,kernel_size=4,bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # 4 x 4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 8 x 8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1,bias=bias),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 16 x 16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1,bias=bias),
                                 # nn.Sigmoid())
                                 nn.Tanh())
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class mnistnet_D(nn.Module):
    def __init__(self, nc=1, ndf=64, BN=True, bias=False): # 128 ok
        super(mnistnet_D,self).__init__()
        if BN:
            # 32 x 32
            self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.BatchNorm2d(ndf),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 16 x 16
            self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.BatchNorm2d(ndf*2),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 8 x 8
            self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.BatchNorm2d(ndf*4),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 4 x 4
            self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=bias))#,
                                     # nn.Sigmoid())
        else:
            # 32 x 32
            self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 16 x 16
            self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 8 x 8
            self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 4 x 4
            self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=bias))#,
        self.apply(weights_init)            

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out.view(-1)

class mnistnet_DSN(nn.Module):
    def __init__(self, nc=1, ndf=64, BN=True, bias=False): # 128 ok
        super(mnistnet_DSN,self).__init__()
        if BN:
            # 32 x 32
            self.layer1 = nn.Sequential(SNConv2d(nc,ndf,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.BatchNorm2d(ndf),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 16 x 16
            self.layer2 = nn.Sequential(SNConv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.BatchNorm2d(ndf*2),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 8 x 8
            self.layer3 = nn.Sequential(SNConv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.BatchNorm2d(ndf*4),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 4 x 4
            self.layer4 = nn.Sequential(SNConv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=bias))#,
                                     # nn.Sigmoid())
        else:
            # 32 x 32
            self.layer1 = nn.Sequential(SNConv2d(nc,ndf,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 16 x 16
            self.layer2 = nn.Sequential(SNConv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 8 x 8
            self.layer3 = nn.Sequential(SNConv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 4 x 4
            self.layer4 = nn.Sequential(SNConv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=bias))#,
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out.view(-1)



class LINnet_G(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100, bias=False): # 256 ok
        super(LINnet_G,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*8,kernel_size=(3, 5), bias=bias),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.ReLU())
        # 3 x 5
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # 6 x 10
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 12 x 20
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1, bias=bias),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 24 x 40
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1, bias=bias),
                                 # nn.Sigmoid())
                                 nn.Tanh())
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class LINnet_D(nn.Module):
    def __init__(self,nc=1,ndf=64,BN=True,bias=False): # 128 ok
        super(LINnet_D,self).__init__()
        # 48 x 80
        if BN:
            self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.BatchNorm2d(ndf),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 24 x 40
            self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.BatchNorm2d(ndf*2),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 12 x 20
            self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.BatchNorm2d(ndf*4),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 6 x 10
            self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.BatchNorm2d(ndf*8),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 3 x 5
            self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(3, 5),stride=1,padding=0,bias=bias))#,
                                     # nn.Sigmoid())
        else:
            self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 24 x 40
            self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 12 x 20
            self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 6 x 10
            self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=bias),
                                     nn.LeakyReLU(0.2,inplace=True))
            # 3 x 5
            self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(3, 5),stride=1,padding=0,bias=bias))#,
                                     # nn.Sigmoid())
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out.view(-1)
