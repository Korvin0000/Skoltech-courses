from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import datetime
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)

from torchvision.utils import save_image
import datasets
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--sde", type=str, default="VESDE", choices=["vesde", 'vpsde', 'subvpsde'])
parser.add_argument("--predictor", type=str, default="ReverseDiffusionPredictor", choices=[
    "ReverseDiffusionPredictor", "AncestralSamplingPredictor", "None"
  ]
)
parser.add_argument("--corrector", type=str, default="LangevinCorrector", choices=[
    "LangevinCorrector", "AnnealedLangevinDynamics", "None"
  ]
)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_scales", type=int, default=1000)
parser.add_argument("--n_steps", type=int, default=1)
parser.add_argument("--data_output_dir", type=str, default="data")
parser.add_argument("--n_images", type=int, default=1024)
args = parser.parse_args()

sde = args.sde.lower()
if sde.lower() == 'vesde':
  from configs.ve import cifar10_ncsnpp_continuous as configs
  ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=args.num_scales)
  sampling_eps = 1e-5
elif sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs  
  ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=args.num_scales)
  sampling_eps = 1e-3
elif sde.lower() == 'subvpsde':
  from configs.subvp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
  config = configs.get_config()
  sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=args.num_scales)
  sampling_eps = 1e-3

batch_size = args.batch_size
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = args.seed 
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
print(config.device)
state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())


#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
if args.predictor == "ReverseDiffusionPredictor":
  predictor = ReverseDiffusionPredictor
elif args.predictor == "AncestralSamplingPredictor":
  predictor = AncestralSamplingPredictor
elif args.predictor == "None":
  predictor = None 

if args.corrector == "LangevinCorrector":
  corrector = LangevinCorrector
elif args.corrector == "AnnealedLangevinDynamics":
  corrector = AnnealedLangevinDynamics
elif args.corrector == "None":
  corrector = None 

snr = 0.16 #@param {"type": "number"}
n_steps = args.n_steps #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)


os.makedirs(args.data_output_dir, exist_ok=True)
save_dir = os.path.join(
  args.data_output_dir, 
  f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{args.sde.lower()}-{args.predictor}-{args.corrector}-n_steps_{args.n_steps}-num_scales_{args.num_scales}"
)
os.makedirs(save_dir)

for batch_start_id_pic in range(0, args.n_images, args.batch_size):
  print("Processed:", batch_start_id_pic, "/", args.n_images)
  x, n = sampling_fn(score_model)
  for i, pic in enumerate(x):
    save_image(pic, os.path.join(save_dir, f"{str(i + batch_start_id_pic).zfill(5)}.png"))
