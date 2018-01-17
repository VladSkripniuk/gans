import torch
from torch.autograd import Variable
import torch.utils.data

from gan import GAN_base


class WGANGP(GAN_base):
    def __init__(self, netG, netD, optimizerD, optimizerG, opt):
        GAN_base.__init__(self, netG, netD, optimizerD, optimizerG, opt)
        self.wgangp_lambda = opt.wgangp_lambda

    def compute_gradient_penalties(self, netD, real_data, fake_data):
        # this code is base on https://github.com/caogang/wgan-gp

        # equalize batch sizes
        
        batch_size = min(real_data.size(0), fake_data.size(0))
        real_data = real_data[:batch_size]
        fake_data = fake_data[:batch_size]
        # get noisy inputs
        eps = torch.rand(batch_size)
        while eps.dim() < real_data.dim():
            eps = eps.unsqueeze(-1)
    
        eps = eps.cuda() if real_data.is_cuda else eps
        interpolates = eps * real_data + (1 - eps) * fake_data
        interpolates = Variable(interpolates, requires_grad=True)

        # push thorugh network
        D_interpolates = netD(interpolates)

        # compute the gradients
        grads = torch.ones(D_interpolates.size())
        grads = grads.cuda() if real_data.is_cuda else grads
        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates, grad_outputs=grads,
                                        create_graph=True, only_inputs=True)

        gradient_input = gradients[0].view(batch_size, -1)
        

        # compute the penalties
        gradient_penalties = (gradient_input.norm(2, dim=1) - 1) ** 2

        return gradient_penalties

    def compute_disc_score(self, data_a, data_b):
        if self.opt.conditional:
            data_a = self.join_xy(data_a)
            data_b = self.join_xy(data_b)

        scores_a = self.netD(data_a)
        scores_b = self.netD(data_b)
        gradient_penalties = self.compute_gradient_penalties(self.netD, data_a.data, data_b.data)

        mean_dim = 0 if scores_a.dim() == 1 else 1
        gradient_penalty = gradient_penalties.mean(mean_dim)
        errD = scores_a.mean(mean_dim) - scores_b.mean(mean_dim) + self.wgangp_lambda * gradient_penalty

        return errD

    def compute_gen_score(self, data):
        if self.opt.conditional:
            data = self.join_xy(data)
        return self.netD(data).mean()
