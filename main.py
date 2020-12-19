from networks import *
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

# path parameters
DATA_DIR = 'data0/lsun/bedroom/0/0/'
SAVE_DIR = 'weights/tuned/lsun/'
CKPT = 'transfer_model/unconditional/bedroom/wgan-gp.model'
SAMPLES_DIR = 'samples/'

# model parameters
N_PIXELS = 64 #length, width of image
BATCH_SIZE = 16
DIM = 64 #model dimensionality
LATENT_DIM = 128
OUTPUT_DIM = N_PIXELS * N_PIXELS * 3 #number of pixels in each image
BN_G = True #whether to include batch normalization
BN_D = True

#training parameters
ITER_START = 0
ITERS = 2
CRITIC_ITERS = 5
GP_WEIGHT = 10
SAVE_SAMPLES_STEP = 1
CHECKPOINT_STEP = 1
LR_G = 0.00001
LR_D = 0.00001
BETA1_G = 0.0
BETA1_D = 0.0
N_GPUS = 1
N_SAMPLES = 10
DEVICES = ['gpu:{}'.format(i) for i in range(N_GPUS)]

# load the data
dataset = loader.FlatFolderDataset(DATA_DIR, loader.train_transform)
data_iter = iter(DataLoader(
    dataset, batch_size = BATCH_SIZE,
    sampler=InfiniteSamplerWrapper(dataset), num_workers=0))
print('loaded dataset')

generator = Generator(DIM, LATENT_DIM, N_PIXELS, BN_G)
critic = Discriminator(DIM, N_PIXELS, BN_D)
print('initialized model')

tf_path = os.path.abspath(CKPT)
init_vars = tf.train.list_variables(tf_path) 
vars_g = {}
vars_d = {}

for name, shape in init_vars:
    array = tf.train.load_variable(tf_path, name)
    name = name.split('/')[-1]
    if name.startswith('Generator'):
        vars_g[name] = torch.from_numpy(array)
    elif name.startswith('Discriminator'):
        vars_d[name] = torch.from_numpy(array)
    else:
        raise Exception('invalid weight name')

utils.load_weights(generator, vars_g)
utils.load_weights(critic, vars_d)
print('loaded weights')

# setup optimizers
G_optimizer = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1_G, 0.9)) #what is colocate_gradients_with_ops
D_optimizer = optim.Adam(critic.parameters(), lr=LR_D, betas= (BETA1_D, 0.9))

fixed_noise = torch.randn((N_SAMPLES, 128), dtype=torch.float32)

def generate_image(iteration):
    samples = generator(N_SAMPLES, noise=fixed_noise)
    samples = ((samples +1.0)*(255 / 2.0))
    return samples.view((N_SAMPLES, 3, N_PIXELS, N_PIXELS))

# restore weights in the model... later

# train
trainer = Trainer(generator, critic, G_optimizer, D_optimizer,
                  gp_weight=GP_WEIGHT, critic_iterations=CRITIC_ITERS, 
                  use_cuda=torch.cuda.is_available())
trainer.train(data_iter, ITERS, save_training_gif=True)

'''
if iteration % CHECKPOINT_STEP == 0:
    utils.save_weights(
        generator, '{}generator_{}.pth.tar'.format(SAVE_DIR, iteration))
    utils.save_weights(
        critic, '{}discriminator_{}.pth.tar'.format(SAVE_DIR, iteration))
'''

