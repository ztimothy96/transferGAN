from networks import *
import time
import torch
import os
import numpy as np
import tensorflow as tf
import utils
import torch.optim as optim
import loader
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
OUTPUT_DIM = N_PIXELS * N_PIXELS * 3 #number of pixels in each image
BN_G = True #whether to include batch normalization
BN_D = True

#training parameters
ITER_START = 0
ITERS = 2
CRITIC_ITERS = 0
LAMBDA = 10 #gradient penalty weight
SAVE_SAMPLES_STEP = 1
CHECKPOINT_STEP = 1
LR_G = 0.00001
LR_D = 0.00001
BETA1_G = 0.0
BETA1_D = 0.0
N_GPUS = 1
N_SAMPLES = 1
DEVICES = ['gpu:{}'.format(i) for i in range(N_GPUS)]


generator = Generator(DIM, N_PIXELS, bn=BN_G)
print('initialized model')

tf_path = os.path.abspath(CKPT)
init_vars = tf.train.list_variables(tf_path) 
vars_g = []

for name, shape in init_vars:
    array = tf.train.load_variable(tf_path, name)
    if name.startswith('Generator'):
        vars_g.append((name, array))

utils.load_weights(generator, vars_g)
print('loaded weights')

samples = generator(N_SAMPLES)
samples = (samples+1.0)/2
samples = samples.view((N_SAMPLES, 3, N_PIXELS, N_PIXELS))
print('generated image')
save_image(samples, SAMPLES_DIR + 'test_samples.png')

