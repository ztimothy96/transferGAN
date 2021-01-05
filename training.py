# https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
# I really like their modularization

import imageio
import numpy as np
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
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
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.save_every = save_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
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

        prob_interpolated = self.D(interpolated)
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
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_iter(self, data_iter, i):
        data = data_iter.next()
        # each iteration trains generator once and critic [critic_iterations] many times
        for _ in range(self.critic_iterations):
            self._critic_train_iteration(data)
        self._generator_train_iteration(data)

        if i+1 % self.print_every == 0:
            print("Iteration {}".format(i+1))
            print("D: {}".format(self.losses['D'][-1]))
            print("GP: {}".format(self.losses['GP'][-1]))
            print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
            print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_iter, n_iters, n_samples=64, iter_start=0,
              save_training_gif=True, save_weights_dir='./', samples_dir='./'):

        fixed_latents = Variable(self.G.sample_latent(n_samples))
        if self.use_cuda:
            fixed_latents = fixed_latents.cuda()
            
        def make_sample_grid():
            samples = self.G(n_samples, noise=fixed_latents)
            samples = (samples+1.0)*(255/2.0)
            img_grid = make_grid(samples.cpu().data)
            # transpose axes to fit imageio convention
            # i.e. (width, height, channels)
            img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
            return img_grid
        
        if save_training_gif:
            training_progress_images = [make_sample_grid()]

        for i in range(iter_start, iter_start + n_iters):
            self._train_iter(data_iter, i)
            if (i+1) % self.save_every == 0:
                print('saving!')
                save_path_g = '{}{}_iter_{}.pt'.format(save_weights_dir, self.G.name, i+1)
                save_path_d = '{}{}_iter_{}.pt'.format(save_weights_dir, self.D.name, i+1)
                torch.save(self.G.state_dict(), save_path_g)
                torch.save(self.D.state_dict(), save_path_d)
                
                # delete previous checkpoints to save space on cluster
                for file in os.listdir(save_weights_dir):
                    path = save_weights_dir + file
                    if file.endswith('.pt') and path not in [save_path_g, save_path_d]:
                        os.remove(path)

                if save_training_gif:
                    training_progress_images.append(make_sample_grid())
                
        if save_training_gif:
            samples_path = '{}training_{}_iters.gif'.format(samples_dir, iter_start + n_iters)
            imageio.mimsave(samples_path, training_progress_images)

        self.plot_losses(samples_dir)

    def sample_generator(self, n_samples):
        latent_samples = Variable(self.G.sample_latent(n_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(n_samples, noise=latent_samples)
        return generated_data

    def plot_losses(self, samples_dir):
        n_iters = len(self.losses['G'])
        x = np.arange(0, n_iters)
        fig, axes = plt.subplots(2, 2)
        ((ax_G, ax_D), (ax_GP, ax_grad_norm)) = axes
        ax_G.plot(x, np.array(self.losses['G']))
        ax_G.set_title('Generator loss')
        ax_D.plot(x, np.array(self.losses['D'])[
            self.critic_iterations-1 : self.critic_iterations*n_iters : self.critic_iterations])
        ax_D.set_title('Discriminator loss')
        ax_GP.plot(x, np.array(self.losses['GP'])[
            self.critic_iterations-1 : self.critic_iterations*n_iters : self.critic_iterations])
        ax_GP.set_title('Gradient penalty')
        ax_grad_norm.plot(x, np.array(self.losses['gradient_norm'])[
            self.critic_iterations-1 : self.critic_iterations*n_iters : self.critic_iterations])
        ax_grad_norm.set_title('Gradient norm')
        for ax in axes.flat:
            ax.set(xlabel='Iterations', ylabel='Loss')
        plt.tight_layout()
        fig.savefig('{}losses.jpg'.format(samples_dir))


# Maybe should have added more options directly to Trainer
# But subclasses are easy to maintain so I'll put it here for now
class EWCTrainer(Trainer):
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, ewc_weight=5e8, critic_iterations=5,
                 print_every=50, save_every=50, use_cuda=False):
        super().__init__(generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight, critic_iterations, print_every, save_every, use_cuda)
        self.ewc_weight = ewc_weight
        # clone init params and save fisher information before training
        self.init_params = [p.clone() for p in generator.parameters()]

        def get_fisher_info(n_samples=30):
            n_params = len(list(generator.parameters()))
            #looping so we don't run out of CUDA memory
            sums = [torch.zeros(p.shape).cuda() if self.use_cuda
                    else torch.zeros(p.shape) for p in generator.parameters()]
            
            for i in range(n_samples):
                sampled_data = self.sample_generator(1)
                log_probs = self.D(sampled_data)
                loss_grads = torch_grad(outputs=log_probs,
                                        inputs=list(generator.parameters()))
                for j in range(n_params):
                    sums[j] = sums[j] + loss_grads[j]**2

            return [s / n_samples for s in sums]
            
        self.fisher = get_fisher_info()

    def _ewc_loss(self, params):
        loss = 0
        for i in range(len(params)):
            loss += torch.sum(self.fisher[i] * (params[i] - self.init_params[i])**2)
        return self.ewc_weight * loss

    def _generator_train_iteration(self, data):
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        ewc_loss = self._ewc_loss(list(self.G.parameters()))
        g_loss = -d_generated.mean() + ewc_loss
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())
