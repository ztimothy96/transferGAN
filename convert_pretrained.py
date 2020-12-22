import torch
import numpy as np
import tensorflow as tf
import argparse
import re
from networks import *

# load pretrained weights from tf to pytorch model
# adapted from https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
# weight_dict is a dictionary of name: tensor
def load_weights(model, weight_dict):
    for name, tensor in weight_dict.items():
        name = name.split('/')[-1]
        name = name.split('.')
        if name[0] in ['Adam', 'Adam_1'] :
            continue 
        pointer = model
        transpose_axes = None

        # print("initialize PyTorch weight {}".format(name))        
        # iterate along the scopes and move pointer accordingly
        for m_name in name[1:]:
            # handle digits in module list
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]

            # Convert parameters final names to pytorch equivalents
            if l[0] == 'Filters' or l[0] == 'W':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'Biases' or l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            
            elif l[0] == 'scale':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'offset':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'moving_mean':
                pointer = getattr(pointer, 'running_mean')
            elif l[0] == 'moving_variance':
                pointer = getattr(pointer, 'running_var')
            
            else:
                pointer = getattr(pointer, l[0])

            # sometimes tf and pytorch conventions are transposed
            if l[0] == 'W':
                transpose_axes = (1, 0)
            if l[0] == 'Filters':
                transpose_axes = (3, 2, 0, 1)
                # transpose_axes = (3, 2, 1, 0)
                # tensorflow conv filter format: H, W, C_in, C_out
                # pytorch conv filter format: C_out, C_in, H, W
                # or maybe it's C_out, C_in, W, H?
                # I'm not sure what the kernel_size[0, 1] represent

            # If module list, access the sub-module with the right number
            if len(l) >= 2:
                num = int(l[1])-1 # our torch indexes start at 0, but TF names start at 1
                pointer = pointer[num]
                
        if transpose_axes:
            tensor = tensor.permute(transpose_axes)
        try:
            assert pointer.shape == tensor.shape
        except AssertionError as e:
            e.args += (pointer.shape, tensor.shape)
            raise
        pointer.data = tensor


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_dir', type=str,
                    default='./pretrained_tf/unconditional/imagenet/wgan-gp.model',
                    help='Directory path to the pretrained TF weights')
parser.add_argument('--save_dir', type=str,
                    default='./pretrained_torch/unconditional/imagenet/',
                    help='Directory to save the converted pytorch weights')
parser.add_argument('--dataset_name', type=str,
                    default='imagenet',
                    help='Dataset on which model was pretrained')

# model parameters
# these are fixed, unless you want to train from scratch...
parser.add_argument('--n_pixels', type=int, default=64,
                    help='Height and width of image')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dim', type=int, default=64,
                    help='Depth of model channels')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='Dimension of latent space')
parser.add_argument('--bn_g', type=bool, default=True,
                    help='Option to include normalization in generator')
parser.add_argument('--bn_d', type=bool, default=True,
                    help='Option to include normalization in discriminator')
args = parser.parse_args()

generator = Generator(args.dim, args.latent_dim, args.n_pixels, args.bn_g)
discriminator = Discriminator(args.dim, args.n_pixels, args.bn_d)
print('initialized model')

init_vars = tf.train.list_variables(args.pretrained_dir) 
weights_g = {}
weights_d = {}

for name, shape in init_vars:
    array = tf.train.load_variable(args.pretrained_dir, name)
    if name.startswith('Generator'):
        weights_g[name] = torch.from_numpy(array)
    elif name.startswith('Discriminator'):
        weights_d[name] = torch.from_numpy(array)

load_weights(generator, weights_g)
load_weights(discriminator, weights_d)
print('loaded weights')

save_path_g = '{}{}_{}.pt'.format(args.save_dir, generator.name, args.dataset_name)
save_path_d = '{}{}_{}.pt'.format(args.save_dir, discriminator.name, args.dataset_name)
torch.save(generator.state_dict(), save_path_g)
torch.save(discriminator.state_dict(), save_path_d)
print('saved pytorch weights')



