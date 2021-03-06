import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn.parameter import Parameter

# note to self: ignoring he initialization throughout, since pre-training.
# https://blog.paperspace.com/pytorch-101-advanced/

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.weight = Parameter(torch.ones(dim))
        self.bias = Parameter(torch.zeros(dim))

    def normalize(self, x, dim=0, eps=0):
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, unbiased=False, keepdim=True)
        return (x-mean)/(var+eps)**0.5

    def affine(self, x):
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def forward(self, x):
        x = self.normalize(x, dim=(1, 2, 3), eps=1e-5)
        return self.affine(x)


        
class ConvMeanPool(nn.Conv2d):
    def __init__(self, input_dim, output_dim, filter_size, bias=True, padding=0):
        super(ConvMeanPool, self).__init__(
            input_dim, output_dim, filter_size, bias=bias, padding=padding)

    def forward(self, x):
        x = super().forward(x)
        x = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2])/4.
        return x


class MeanPoolConv(nn.Conv2d):
    def __init__(self, input_dim, output_dim, filter_size, bias=True, padding=0):
        super(MeanPoolConv, self).__init__(
            input_dim, output_dim, filter_size, bias=bias, padding=padding)

    def forward(self, x):
        x = output = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2])/ 4.
        x = super().forward(x)
        return x


class UpsampleConv(nn.Conv2d):
    def __init__(self, input_dim, output_dim, filter_size,
                 bias=True, padding=0, scale_factor=2, mode='nearest'):
        super(UpsampleConv, self).__init__(
            input_dim, output_dim, filter_size, bias=bias, padding=padding)
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
        
    def forward(self, x):
        x = self.up(x)
        x = super().forward(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, norm='batch',
                 resample=None, bn=False):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        p = math.floor(filter_size / 2) # same padding
        if resample=='down':
            self.Shortcut = MeanPoolConv(input_dim, output_dim, 1)
            self.Conv = nn.ModuleList(
                [nn.Conv2d(input_dim, input_dim, filter_size, padding=p, bias=False),
                 ConvMeanPool(input_dim, output_dim, filter_size, padding=p)])
        elif resample=='up':
            self.Shortcut = UpsampleConv(input_dim, output_dim, 1)
            self.Conv = nn.ModuleList(
                [UpsampleConv(input_dim, output_dim, filter_size, padding=p, bias=False),
                 nn.Conv2d(output_dim, output_dim, filter_size, padding=p)])
        elif resample==None:
            self.Shortcut = nn.Conv2d(input_dim, output_dim, 1)
            self.Conv = nn.ModuleList(
                [nn.Conv2d(input_dim, input_dim, filter_size, padding=p, bias=False),
                 nn.Conv2d(input_dim, output_dim, filter_size, padding=p)])
        else:
            raise Exception('invalid resample value')

        if bn:
            if norm=='batch':
                self.BN = nn.ModuleList([
                    nn.BatchNorm2d(input_dim, eps=1e-3, momentum=1.0),
                    nn.BatchNorm2d(output_dim, eps=1e-3, momentum=1.0)])

            elif norm=='layer':
                '''
                self.BN = nn.ModuleList([
                    nn.LayerNorm(input_dim),
                    nn.LayerNorm(input_dim)])
                '''
                self.BN = nn.ModuleList([
                    LayerNorm(input_dim),
                    LayerNorm(input_dim)])

            else:
                raise Exception('invalid normalization value')
        else:
            self.register_parameter('BN', None)
            
    
    def forward(self, x):
        if self.output_dim == self.input_dim and self.resample==None:
            shortcut = x
        else:
            shortcut = self.Shortcut(x)

        if self.BN:
            x = self.BN[0](x)
            
        x = F.relu(x)
        x = self.Conv[0](x)
        
        if self.BN:
            x =self.BN[1](x)

        x = F.relu(x)
        x = self.Conv[1](x)
        return shortcut + x


class Generator(nn.Module):
    def __init__(self, dim, latent_dim, n_pixels, bn=True, name='generator'):
        super(Generator, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.n_pixels = n_pixels
        self.name=name
        
        fact = n_pixels // 16
        self.Input = nn.Linear(latent_dim, fact*fact*8*dim)
        self.Res = nn.ModuleList([ResBlock(8*dim, 8*dim, 3,
                                           resample='up', bn=bn, norm='batch'),
                                  ResBlock(8*dim, 4*dim, 3,
                                           resample='up', bn=bn, norm='batch'),
                                  ResBlock(4*dim, 2*dim, 3,
                                           resample='up', bn=bn, norm='batch'),
                                  ResBlock(2*dim, 1*dim, 3,
                                           resample='up', bn=bn, norm='batch')])
        if bn:
            self.OutputN = nn.BatchNorm2d(1*dim)
        else:
            self.register_parameter('OutputN', None)
        self.Output = nn.Conv2d(1*dim, 3, 3, padding=1)

    def forward(self, n_samples, noise=None):
        if noise is None:
            noise = self.sample_latent(n_samples)

        ## supports 32x32 images; figure out why magic number 16
        fact = self.n_pixels // 16
        
        x = self.Input(noise)
        x = x.view(-1, 8*self.dim, fact, fact)

        for block in self.Res:
            x = block(x)
            
        if self.OutputN:
            x = self.OutputN(x)

        x = F.relu(x)
        x = self.Output(x)
        x = torch.tanh(x)
        x = x.view(-1, 3, self.n_pixels, self.n_pixels)
        return x

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, dim, n_pixels, bn=True, name='discriminator'):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.n_pixels = n_pixels
        self.name=name
        
        fact = n_pixels // 16
        self.Input = nn.Conv2d(3, dim, 3, padding=1)
        self.Res = nn.ModuleList([ResBlock(1*dim, 2*dim, 3,
                                           resample='down', bn=bn, norm='layer'),
                                  ResBlock(2*dim, 4*dim, 3,
                                           resample='down', bn=bn, norm='layer'),
                                  ResBlock(4*dim, 8*dim, 3,
                                           resample='down', bn=bn, norm='layer'),
                                  ResBlock(8*dim, 8*dim, 3,
                                           resample='down', bn=bn, norm='layer')])
        self.Output = nn.Linear(fact*fact*8*dim, 1)

    def forward(self, x):
        fact = self.n_pixels // 16
        x = x.view(-1, 3, self.n_pixels, self.n_pixels)
        x = self.Input(x)
        for block in self.Res:
            x = block(x)
        x = x.view(-1, fact * fact * 8 * self.dim)
        x = self.Output(x)
        x = x.view(-1)
        return x
    
