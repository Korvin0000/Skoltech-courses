import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from tqdm.auto import tqdm
from torch.distributions import MultivariateNormal
import torch_fidelity
import subprocess
import json 
import pandas as pd
BATCH_SIZE = 10000
LATENT_SIZE = 100
NUM_STEPS = 2001
EPS = 6e-4
DECAY_DELTA = 100
DECAY_RATE = 5
DELTA = 50


num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from dcgan import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

# load weights
D.load_state_dict(torch.load('weights/netD_epoch_199.pth'))   
G.load_state_dict(torch.load('weights/netG_epoch_199.pth'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()


latent_distribution = MultivariateNormal(
    loc = torch.zeros(LATENT_SIZE).cuda(),
    scale_tril=torch.diag(torch.ones(LATENT_SIZE)).cuda()
)
standard_normal = MultivariateNormal(
    loc=torch.zeros(LATENT_SIZE).cuda(), 
    scale_tril=torch.diag(torch.ones(LATENT_SIZE)).cuda()
)

def batch_sample(batch_size, dist):
  return torch.tensor([dist.sample().tolist() for _ in range(batch_size)])


def Energy(z, D, G, latent_distribution):
  fake_images = G(z)
  outputs = D(fake_images)
  d = torch.logit(outputs)
  return -latent_distribution.log_prob(z.squeeze()) - d

def run_langevin_for_batch(batch_noise, num_steps, eps, decay_delta, decay_rate, D, G, standard_normal, latent_distribution):
    D.requires_grad_(False)
    G.requires_grad_(False)
    batch_size = len(batch_noise)
    z = torch.nn.Parameter(batch_noise, requires_grad=True)
    
    pictures = []

    for i in tqdm(range(num_steps)):
        if i % decay_delta == 0:
            eps /= decay_rate ** (i / decay_delta)
            optim = torch.optim.SGD([z], lr=eps/2.)
        energy = Energy(z, D, G, latent_distribution).sum()
        energy.backward()
        optim.step()
        optim.zero_grad()
        with torch.no_grad():     
            z += torch.sqrt(torch.tensor(eps)) * standard_normal.sample_n(batch_size).cuda()[:, :, None, None]
        pictures.append(G(z).detach().cpu())
        
    return pictures
        
        
batch_noise = latent_distribution.sample_n(BATCH_SIZE)[:, :, None, None].cuda()

pics = run_langevin_for_batch(
    batch_noise, 
    num_steps=NUM_STEPS,
    decay_delta=DECAY_DELTA,
    decay_rate=10,
    eps=EPS, 
    D=D, 
    G=G, 
    standard_normal=standard_normal, 
    latent_distribution=latent_distribution
)

fids = [] 
for step in tqdm(range(NUM_STEPS), leave=False):
    if step % DELTA == 0:
        os.makedirs(f'tmp/step_{step}', exist_ok=True)
        for i in tqdm(range(BATCH_SIZE), leave=False):
            save_image(pics[step][i], f'tmp/step_{step}/{i}.png')
        with open(f'tmp/step_{step}/fid.json', 'w') as f:
            fidelity_args = f"fidelity --gpu 1 --fid --input1 tmp/step_{step}/ --input2 cifar10-train --json --silent".split()
            subprocess.call(fidelity_args, stdout=f)
        with open(f'tmp/step_{step}/fid.json') as f:
            fid = json.load(f)['frechet_inception_distance']
        fids.append(fid)
        pd.DataFrame({"fids" : fids}).to_csv("fids_1000.csv")
        plt.clf()
        plt.cla()
        plt.close()
        plt.plot(range(0, DELTA * len(fids), DELTA), fids, c="b")
        plt.title("Progression of Fréchet inception distance")
        plt.xlabel("Langevin dynamics sampling iterations")
        plt.ylabel("Fréchet inception distance (FID)")
        plt.grid()
        plt.savefig("pretty_fids.svg")
        plt.savefig("pretty_fids.png")
