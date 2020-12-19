from networks import *
import argparse
import time
import torch
import os
import numpy as np
import tensorflow as tf
import utils
import torch.optim as optim
import loader
from training import Trainer
from torch.utils.data import DataLoader
from sampler import InfiniteSamplerWrapper

print('imported all libraries')

# need different data types depending on whether we're using GPU or CPU
# use CPU for now, add params later

# parsing based on https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data0/lsun/bedroom/0/0/',
                    help='Directory path to training images')
parser.add_argument('--pretrained_dir',
                    default='./transfer_model/unconditional/bedroom/wgan-gp.model',
                    help='Directory path to the pretrained TF weights')
# not passed along yet...
parser.add_argument('--save_dir', default='./save_weights',
                    help='Directory to save the model')
parser.add_argument('--samples_dir', type=str, default='.samples/',
                    help='Directory to samples images')

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

#training parameters
parser.add_argument('--iter_start', type=int, default=64,
                    help='Iteration from which to resume training')
parser.add_argument('--n_iters', type=int, default=2,
                    help='Number of iterations to train for')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='Number of critic training steps per generator step')
parser.add_argument('--gp_weight', type=float, default=10,
                    help='Regularization weight for the gradient penalty')
parser.add_argument('--save_every', type=int, default=1,
                    help='Number of iterations before saving model weights and samples')
parser.add_argument('--print_every', type=int, default=1,
                    help='Number of iterations before printing losses to console')
parser.add_argument('--lr_g', type=float, default=1e-5,
                    help='Learning rate of generator')
parser.add_argument('--lr_d', type=float, default=1e-5,
                    help='Learning rate of discriminator')
parser.add_argument('--beta1_g', type=float, default=0.0,
                    help='Exponential decay rate for generator 1st moment estimates')
parser.add_argument('--beta1_d', type=float, default=0.0,
                    help='Exponential decay rate for discriminator 1st moment estimates')
parser.add_argument('--n_gpus', type=int, default=1,
                    help='Number of GPUs available')
parser.add_argument('--n_samples', type=int, default=64,
                    help='Number of samples to save at each checkpoint')
args = parser.parse_args()

# load the data
dataset = loader.FlatFolderDataset(args.data_dir, loader.train_transform)
data_iter = iter(DataLoader(
    dataset, batch_size = args.batch_size,
    sampler=InfiniteSamplerWrapper(dataset), num_workers=0))
print('loaded dataset')

generator = Generator(
    args.dim, args.latent_dim, args.n_pixels, args.bn_g)
discriminator = Discriminator(
    args.dim, args.n_pixels, args.bn_d)
print('initialized model')

tf_path = os.path.abspath(args.pretrained_dir)
init_vars = tf.train.list_variables(tf_path) 
weights_g = {}
weights_d = {}

for name, shape in init_vars:
    array = tf.train.load_variable(tf_path, name)
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

# set up optimizers
#what is tf.colocate_gradients_with_ops?
G_optimizer = optim.Adam(generator.parameters(),
                         lr=args.lr_g, betas=(args.beta1_g, 0.9))
D_optimizer = optim.Adam(discriminator.parameters(),
                         lr=args.lr_d, betas= (args.beta1_d, 0.9))

# restore weights in the model from pytorch... later

# train
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  gp_weight=args.gp_weight, critic_iterations=args.critic_iters, 
                  print_every=args.print_every, save_every=args.save_every,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_iter, args.n_iters, save_training_gif=True)
