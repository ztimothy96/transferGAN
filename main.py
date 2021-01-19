from networks import *
import argparse
import torch
import numpy as np
import torch.optim as optim
import time
from torchvision import transforms, utils
from loader import FlatFolderDataset, InfiniteSamplerWrapper
from training import *
from torch.utils.data import DataLoader

# for reproducibility
torch.manual_seed(0)
np.random.seed(42)

# parsing based on https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='../../../data/tzhou28/bedroom00/',
                    # default='./data0/lsun/bedroom/0/0/',
                    help='Directory path to training images')
parser.add_argument('--pretrained_dir_g',
                    default='./pretrained_torch/unconditional/imagenet/generator_imagenet.pt',
                    help='Path to the pretrained pytorch generator weights')
parser.add_argument('--pretrained_dir_d',
                    default='./pretrained_torch/unconditional/imagenet/discriminator_imagenet.pt',
                    help='Path to the pretrained pytorch discriminator weights')
parser.add_argument('--save_dir', default='./save_weights/',
                    help='Directory to save the model')
parser.add_argument('--samples_dir', type=str, default='./samples/',
                    help='Directory to save sample images')

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
parser.add_argument('--mode', type=str, default='',
                    help='Set to "ewc" to use ewc loss')
parser.add_argument('--freeze', type=str, default='',
                    help='Set to "batch" to freeze everything except batchnorm')
parser.add_argument('--iter_start', type=int, default=0,
                    help='Number of iterations previously trained')
parser.add_argument('--n_examples', type=int, default=1000,
                    help='Number of training examples to use')
parser.add_argument('--n_iters', type=int, default=0000,
                    help='Number of additional iterations to train')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='Number of critic training steps per generator step')
parser.add_argument('--gp_weight', type=float, default=10,
                    help='Regularization weight for the gradient penalty')
parser.add_argument('--lr_g', type=float, default=1e-5,
                    help='Learning rate of generator')
parser.add_argument('--lr_d', type=float, default=1e-5,
                    help='Learning rate of discriminator')
parser.add_argument('--beta1_g', type=float, default=0.0,
                    help='Exponential decay rate for generator 1st moment estimates')
parser.add_argument('--beta1_d', type=float, default=0.0,
                    help='Exponential decay rate for discriminator 1st moment estimates')
parser.add_argument('--ewc_weight', type=float, default=5e4,
                    help='Regularization weight for EWC loss')
parser.add_argument('--n_gpus', type=int, default=torch.cuda.device_count(),
                    help='Number of GPUs available')

#checkpoint output parameters
parser.add_argument('--save_every', type=int, default=1000,
                    help='Number of iterations before saving model weights and samples')
parser.add_argument('--print_every', type=int, default=1000,
                    help='Number of iterations before printing losses to console')
parser.add_argument('--n_samples', type=int, default=64,
                    help='Number of samples to save at each checkpoint')
args = parser.parse_args()

# available_gpus = [torch.cuda.device(i) for i in range(args.n_gpus)]
for arg in vars(args):
	print(arg, getattr(args, arg))

# load the data
# data loaded in range [0.0, 1.0] while network output is in range [-1.0, 1.0]
train_transform = transforms.Compose([
    transforms.Resize(size=(args.n_pixels, args.n_pixels)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*2.0 - 1.0)])

dataset = FlatFolderDataset(
    args.data_dir,transform=train_transform, n_examples=args.n_examples)
data_iter = iter(DataLoader(
    dataset, batch_size = args.batch_size,
    sampler=InfiniteSamplerWrapper(dataset), num_workers=0))

# load model and weights
generator = Generator(args.dim, args.latent_dim, args.n_pixels, args.bn_g)
discriminator = Discriminator(args.dim, args.n_pixels, args.bn_d)
generator.load_state_dict(torch.load(args.pretrained_dir_g))
discriminator.load_state_dict(torch.load(args.pretrained_dir_d))
if args.freeze=='batch':
        for name, param in generator.named_parameters():
                if not ('BN' in name or 'OutputN' in name):
                        param.requires_grad = False

# set up optimizers
G_optimizer = optim.Adam(generator.parameters(),
                         lr=args.lr_g, betas=(args.beta1_g, 0.9))
D_optimizer = optim.Adam(discriminator.parameters(),
                         lr=args.lr_d, betas= (args.beta1_d, 0.9))

start = time.time()

# train
if args.mode=='ewc':
        trainer = EWCTrainer(generator, discriminator, G_optimizer, D_optimizer,
                             gp_weight=args.gp_weight, ewc_weight=args.ewc_weight,
                             critic_iterations=args.critic_iters,
                             print_every=args.print_every, save_every=args.save_every,
                             use_cuda=torch.cuda.is_available())
else:
        trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                          gp_weight=args.gp_weight, critic_iterations=args.critic_iters,
                          print_every=args.print_every, save_every=args.save_every,
                          use_cuda=torch.cuda.is_available())

trainer.train(data_iter, args.n_iters, n_samples=args.n_samples,
              iter_start=args.iter_start, save_training_gif=True,
              save_weights_dir=args.save_dir, samples_dir=args.samples_dir)

end = time.time()
print('Time elapsed: {} seconds'.format(end-start))
