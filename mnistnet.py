import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SNConv2d import SNConv2d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
            self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0))#,

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

<<<<<<< HEAD
# improved wgan pytorch

DIM = 64
OUTPUT_DIM = 784

class Generator(nn.Module):
    def __init__(self, nz=100, BN=False):
        super(Generator, self).__init__()

        if BN:
            preprocess = nn.Sequential(
                # nn.Linear(100, 4*4*4*DIM),
                nn.ConvTranspose2d(nz, 4*DIM, 4),
                nn.BatchNorm2d(4*DIM),
                nn.ReLU(True),
            )
            block1 = nn.Sequential(
                nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
                nn.BatchNorm2d(2*DIM),
                nn.ReLU(True),
            )
            block2 = nn.Sequential(
                nn.ConvTranspose2d(2*DIM, DIM, 5),
                nn.BatchNorm2d(DIM),
                nn.ReLU(True),
            )
        else:
            preprocess = nn.Sequential(
                # nn.Linear(100, 4*4*4*DIM),
                nn.ConvTranspose2d(nz, 4*DIM, 4),
                nn.ReLU(True),
            )
            block1 = nn.Sequential(
                nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
                nn.ReLU(True),
            )
            block2 = nn.Sequential(
                nn.ConvTranspose2d(2*DIM, DIM, 5),
                nn.ReLU(True),
            )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 10, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :, :]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.tanh(output)
        #print output.size()
        return output#.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self, nc=1, BN=False):
        super(Discriminator, self).__init__()

        if BN:
            main = nn.Sequential(
                nn.Conv2d(nc, DIM, 5, stride=2, padding=2),
                # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
                nn.BatchNorm2d(DIM),
                nn.ReLU(True),
                nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                nn.BatchNorm2d(2*DIM),
                nn.ReLU(True),
                nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
                nn.BatchNorm2d(4*DIM),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                nn.ReLU(True),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                # nn.LeakyReLU(True),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                # nn.LeakyReLU(True),
            )
        else:
            main = nn.Sequential(
                nn.Conv2d(nc, DIM, 5, stride=2, padding=2),
                # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
                nn.ReLU(True),
                nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                nn.ReLU(True),
                nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                nn.ReLU(True),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                # nn.LeakyReLU(True),
                # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
                # nn.LeakyReLU(True),
            )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        # input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
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

class netG(nn.Module):
    def __init__(self, nc=1, ngf=64, nz=100): # 256 ok
        super(netG,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*8,kernel_size=4),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.ReLU())
        # 4 x 4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*8,ngf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        # 8 x 8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 16 x 16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 16 x 16
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=3,stride=1,padding=1),
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

class netD(nn.Module):
    def __init__(self,nc=1,ndf=64,spec_norm=True,detach=False): # 128 ok
        super(netD,self).__init__()
        # 32 x 32
        self.layer0_0 = nn.Sequential(SNConv2d(nc,ndf,kernel_size=3,stride=1,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))

        self.layer0_1 = nn.Sequential(SNConv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))
        
        self.layer1_0 = nn.Sequential(SNConv2d(ndf*2,ndf*2,kernel_size=3,stride=1,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))
        
        self.layer1_1 = nn.Sequential(SNConv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))
        
        self.layer2_0 = nn.Sequential(SNConv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))
        
        self.layer2_1 = nn.Sequential(SNConv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))

        self.layer3_0 = nn.Sequential(SNConv2d(ndf*8,ndf*8,kernel_size=3,stride=1,padding=1,spec_norm=spec_norm,detach=detach),
                                 nn.LeakyReLU(0.1,inplace=True))

        self.layer3_1 = nn.Sequential(SNConv2d(ndf*8,1,kernel_size=4,stride=1,padding=0,spec_norm=spec_norm,detach=detach),
                                    )
        # self.layer0_0 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))

        # self.layer0_1 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))
        
        # self.layer1_0 = nn.Sequential(nn.Conv2d(ndf*2,ndf*2,kernel_size=3,stride=1,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))
        
        # self.layer1_1 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))
        
        # self.layer2_0 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))
        
        # self.layer2_1 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))

        # self.layer3_0 = nn.Sequential(nn.Conv2d(ndf*8,ndf*8,kernel_size=3,stride=1,padding=1),
        #                          nn.LeakyReLU(0.1,inplace=True))

        # self.layer3_1 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0),
        #                             )
        
        self.apply(weights_init)
            

    def forward(self,x):
        out = x
        out = self.layer0_0(out)
        out = self.layer0_1(out)
        out = self.layer1_0(out)
        out = self.layer1_1(out)
        out = self.layer2_0(out)
        out = self.layer2_1(out)
        out = self.layer3_0(out)
        out = self.layer3_1(out)
        return out.view(-1)

