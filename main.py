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
N_SAMPLES = 10
DEVICES = ['gpu:{}'.format(i) for i in range(N_GPUS)]

# load the data
dataset = loader.FlatFolderDataset(DATA_DIR, loader.train_transform)
dataloader = DataLoader(
    dataset,batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
data_iter = iter(dataloader)
print('loaded dataset')

generator = Generator(DIM, N_PIXELS, BN_G)
critic = Discriminator(DIM, N_PIXELS, BN_D)
print('initialized model')

tf_path = os.path.abspath(CKPT)
init_vars = tf.train.list_variables(tf_path) 
vars_g = []
vars_d = []

for name, shape in init_vars:
    array = tf.train.load_variable(tf_path, name)
    if name.startswith('Generator'):
        vars_g.append((name, array))
    elif name.startswith('Discriminator'):
        vars_d.append((name, array))
    else:
        raise Exception('invalid weight name')

utils.load_weights(generator, vars_g)
utils.load_weights(critic, vars_d)
print('loaded weights')

# setup optimizers
gen_train_op = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1_G, 0.9)) #what is colocate_gradients_with_ops
disc_train_op = optim.Adam(critic.parameters(), lr=LR_D, betas= (BETA1_D, 0.9))

fixed_noise = torch.randn((N_SAMPLES, 128), dtype=torch.float32)

def generate_image(iteration):
    samples = generator(N_SAMPLES, noise=fixed_noise)
    samples = ((samples +1.0)*(255 / 2.0))
    return samples.view((N_SAMPLES, 3, N_PIXELS, N_PIXELS))

# restore weights in the model... later

# training loop
for t in range(ITERS):
    iteration = t + ITER_START
    start_time = time.time()

    gen_costs, disc_costs = [], []
    if iteration > 0:
        real_images = next(data_iter)
        split_real_images = torch.split(real_images, len(DEVICES))
        for idx, (device, real_image) in enumerate(
            zip(DEVICES, split_real_images)):
            
            real_data = 2*(real_image.type(torch.FloatTensor)/255.0 - 0.5)
            real_data = real_data.view((BATCH_SIZE//len(DEVICES),  OUTPUT_DIM))       
            fake_data = generator(BATCH_SIZE//len(DEVICES))

            disc_real = critic(real_data)
            disc_fake = critic(fake_data)

            gen_cost = -torch.mean(disc_fake)
            gen_costs.append(gen_cost)

        gen_cost = sum(gen_costs) / len(DEVICES)

        #train critic
        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            disc_wgan_pure = torch.mean(disc_fake) - torch.mean(disc_real) # vanilla Wasserstein cost

            #sample along the straight lines between real, fake distributions
            alpha = torch.rand((BATCH_SIZE//len(DEVICES), 1))
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            outputs = critic(interpolates)

            # seems like overkill to pass through whole graph?
            # https://discuss.pytorch.org/t/directly-getting-gradients/688
            # try autograd instead?
            gradients = torch.autograd.grad(
                outputs.split(1), interpolates, only_inputs=True)
            print(gradients)
            slopes = torch.sqrt(sum(g**2 for g in gradients))
            gradient_penalty = torch.mean((slopes-1.0)**2)
            # don't forget to zero gradients out at some point
            
            disc_wgan = disc_wgan_pure + LAMBDA*gradient_penalty
            disc_cost = disc_wgan
            disc_costs.append(disc_cost)
            disc_cost = sum(disc_costs) / len(DEVICES)

    # plot costs

    # save sample images and weights
    if iteration < 100 or iteration % SAVE_SAMPLES_STEP == 0:
        samples = generate_image(iteration)
        utils.save_images(samples, SAMPLES_DIR+'samples_{}.png'.format(iteration))

    if iteration % CHECKPOINT_STEP == 0:
        utils.save_weights(
            generator, '{}generator_{}.pth.tar'.format(SAVE_DIR, iteration))
        utils.save_weights(
            critic, '{}discriminator_{}.pth.tar'.format(SAVE_DIR, iteration))


