import torch
import numpy as np
import tensorflow as tf
import utils
import argparse
from networks import *

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_dir', type=str,
                    default='./pretrained_tf/unconditional/bedroom/wgan-gp.model',
                    help='Directory path to the pretrained TF weights')
parser.add_argument('--save_dir', type=str,
                    default='./pretrained_torch/unconditional/bedroom/',
                    help='Directory to save the converted pytorch weights')
parser.add_argument('--dataset_name', type=str,
                    default='bedroom',
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
    name = name.split('/')[-1]
    if name.startswith('Generator'):
        weights_g[name] = torch.from_numpy(array)
    elif name.startswith('Discriminator'):
        weights_d[name] = torch.from_numpy(array)
    else:
        raise Exception('invalid weight name')

utils.load_weights(generator, weights_g)
utils.load_weights(discriminator, weights_d)
print('loaded weights')

save_path_g = '{}{}_{}.pt'.format(args.save_dir, generator.name, args.dataset_name)
save_path_d = '{}{}_{}.pt'.format(args.save_dir, discriminator.name, args.dataset_name)
torch.save(generator.state_dict(), save_path_g)
torch.save(discriminator.state_dict(), save_path_d)
print('saved pytorch weights')



