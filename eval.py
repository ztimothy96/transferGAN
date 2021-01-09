import numpy as np
import torch
import os
import imageio
import argparse
from networks import Generator

# for reproducibility
torch.manual_seed(0)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,
                    help='Path to the pytorch generator weights')
parser.add_argument('--n_samples', type=int, default=1000,
                    help='Number of sample images to generate')
parser.add_argument('--save_dir', default='./gen_images/',
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
generator = Generator(args.dim, args.latent_dim, args.n_pixels, args.bn_g)
generator.load_state_dict(torch.load(args.model_dir, map_location=device))
generator.eval()

with torch.no_grad():
    for i in range(args.n_samples):
        samples = generator(1)
        samples = (samples+1.0)*(255/2.0)
        img = np.transpose(samples.numpy()[0], (1, 2, 0))
        imageio.imwrite('{}{}{}'.format(args.save_dir, i+1, args.file_type), img)
