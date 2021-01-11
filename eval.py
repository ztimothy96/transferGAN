import numpy as np
import torch
import os
import imageio
import argparse
from networks import Generator
from torchvision import transforms, utils
from loader import FlatFolderDataset, InfiniteSamplerWrapper
from torch.utils.data import DataLoader

'''
this file creates datasets from real and generated images
used to compute FID score
'''

# for reproducibility
torch.manual_seed(0)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data0/lsun/bedroom/0/0/',
                    help='Path to the training images')
parser.add_argument('--model_dir', type=str, default='../../tzhou_shared/results/output_1000/generator_iter_9000.pt',
                    help='Path to the pytorch generator weights')
parser.add_argument('--n_samples', type=int, default=1000,
                    help='Number of sample images to generate')
parser.add_argument('--gen_save_dir', default='./gen_images/',
                    help='Directory to save the generated images')
parser.add_argument('--real_save_dir', default='./real_images/',
                    help='Directory to save the generated images')
parser.add_argument('--file_type', default='.jpg',
                    help='Directory to save the generated images')

# model parameters
parser.add_argument('--n_pixels', type=int, default=64,
                    help='Height and width of image')
parser.add_argument('--dim', type=int, default=64,
                    help='Depth of model channels')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='Dimension of latent space')
parser.add_argument('--bn_g', type=bool, default=True,
                    help='Option to include normalization in generator')
args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def make_generated_dataset():
    generator = Generator(args.dim, args.latent_dim, args.n_pixels, args.bn_g)
    generator.load_state_dict(torch.load(args.model_dir, map_location=device))
    generator.eval()

    with torch.no_grad():
        for i in range(args.n_samples):
            samples = generator(1)
            samples = (samples+1.0)*(255/2.0)
            img = np.transpose(samples.numpy()[0], (1, 2, 0))
            imageio.imwrite('{}{}{}'.format(args.gen_save_dir, i+1, args.file_type), img)

def make_real_dataset():
    train_transform = transforms.Compose([
        transforms.Resize(size=(args.n_pixels, args.n_pixels)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2.0 - 1.0)])

    dataset = FlatFolderDataset(
        args.data_dir,transform=train_transform, n_examples=args.n_samples)
    data_iter = iter(DataLoader(
        dataset, batch_size = 1,
        sampler=InfiniteSamplerWrapper(dataset), num_workers=0))
    for i in range(args.n_samples):
        samples = data_iter.next()
        img = np.transpose(samples.numpy()[0], (1, 2, 0))
        imageio.imwrite('{}{}{}'.format(args.real_save_dir, i+1, args.file_type), img)
    

make_real_dataset()
make_generated_dataset()



