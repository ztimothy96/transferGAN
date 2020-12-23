# https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
# I really like their modularization

import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5,
                 print_every=50, save_every=50,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.save_every = save_every

        if self.use_cuda:
            print('using cuda')
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())
        

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_iter(self, data_iter, i):
        data = data_iter.next()
        self.num_steps += 1
        print('Starting critic iteration {}'.format(i+1))
        self._critic_train_iteration(data)
        # Only update generator every |critic_iterations| iterations
        if self.num_steps % self.critic_iterations == 0:
            print('Starting generator iteration {}'.format(i+1))
            self._generator_train_iteration(data)

        if i % self.print_every == 0:
            print("Iteration {}".format(i + 1))
            print("D: {}".format(self.losses['D'][-1]))
            print("GP: {}".format(self.losses['GP'][-1]))
            print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
            if self.num_steps > self.critic_iterations:
                print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_iter, n_iters, n_samples=64,
              save_training_gif=True, save_weights_dir='./', samples_dir='./'):
        if save_training_gif:
            fixed_latents = Variable(self.G.sample_latent(n_samples))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for i in range(n_iters):
            self._train_iter(data_iter, i)

            if i % self.save_every == 0:
                save_path_g = '{}{}_iter_{}.pt'.format(save_weights_dir, self.G.name, i)
                save_path_d = '{}{}_iter_{}.pt'.format(save_weights_dir, self.D.name, i)
                torch.save(self.G.state_dict(), save_path_g)
                torch.save(self.D.state_dict(), save_path_d)

                if save_training_gif:
                    samples = self.G(n_samples, noise=fixed_latents)
                    samples = (samples+1.0)*(255/2.0)
                    img_grid = make_grid(samples.cpu().data)
                    # transpose axes to fit imageio convention
                    # i.e. (width, height, channels)
                    img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                    training_progress_images.append(img_grid)
                
        if save_training_gif:
            samples_path = '{}training_{}_iters.gif'.format(samples_dir, n_iters)
            imageio.mimsave(samples_path, training_progress_images)

    def sample_generator(self, n_samples):
        latent_samples = Variable(self.G.sample_latent(n_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(n_samples, noise=latent_samples)
        return generated_data

    def sample(self, n_samples):
        generated_data = self.sample_generator(n_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
