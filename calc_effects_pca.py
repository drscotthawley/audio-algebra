#!/usr/bin/env python3

# Author: Scott H. Hawley 

# Sample usage:
# $ CUDA_VISIBLE_DEVICES=7 ./calc_effects_pca.py --config-file bdct-chunk-pca.ini --batch-size=256 --load-frac=1.0

from prefigure.prefigure import get_all_args, push_wandb_config
import math
import json
import subprocess
import os, sys
import random
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import torch
import torchaudio
from torch import optim, nn, utils, Tensor
from torch.nn import functional as F

from tqdm.auto import tqdm, trange
from einops import rearrange, repeat

import wandb

from aeiou.core import get_device
from aeiou.viz import *
from aeiou.hpc import freeze
from aeiou.datasets import AudioDataset
from audio_algebra.datasets import DualEffectsDataset
from audio_algebra.aa_effects import *
from audio_algebra.given_models import SpectrogramAE, MagSpectrogramAE, MelSpectrogramAE, DVAEWrapper
from audiomentations import *   # list of effects


def sorted_eig(cov):  # For now we sort 'by convention'. For PCA the sorting is key.
    lambdas, vs = torch.linalg.eigh(cov)   # cov is symmetric so we can use eigh instead of eig
    lambdas, indices = torch.sort(lambdas, dim=0, descending=True)
    vs = torch.index_select(vs, 0, indices)
    return lambdas, vs


### MAIN ### 
def main(): 
    args = get_all_args()
    print("args = ",args)

    device = get_device()
    print('Using device:', device)

    print("Setting up dataset")
    effects_list = [Gain, BandPassFilter, BandStopFilter, HighPassFilter, LowPassFilter]
    train_set = AudioDataset([os.path.expanduser(args.training_dir)], load_frac=args.load_frac)
    #train_set = DualEffectsDataset([os.path.expanduser(args.training_dir)], load_frac=args.load_frac, effects_list=effects_list) 
    train_dl = utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                    num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    print("Dataset ready to go! length =",len(train_set))

    debug=False

    # init the given autoencoder
    with torch.no_grad():
        given_model = DVAEWrapper()
        given_model.setup(gdrive=False)
        given_model.to(device)
        print("Given model is ready to go!")
    
        print("starting wandb")
        wandb.init(project='aa-dvae-pca', entity="harmonai")
    
        step, npoints = 0, 0
        cov_numerator = None
        for bi, x_batch in enumerate(tqdm(train_dl)): 
            step += 1
            x_batch = x_batch.to(device)
            ys = given_model.encode(x_batch).detach()
            if debug: print(f"\nbi={bi}: ys.shape = {ys.shape}",flush=True)
            ys = rearrange(ys, 'b d n -> d (b n)')  # torch.cov expects variables on rows
            npoints += ys.shape[1] # running count of how many points 
            if cov_numerator is None:
                cov_numerator = torch.cov(ys)*(ys.shape[1]-1)
            else:
                cov_numerator += torch.cov(ys)*(ys.shape[1]-1)
    
            cov = cov_numerator / (npoints-1)   # running covariance matric
            lambdas, vs = sorted_eig(cov)
            if debug: print("lambdas.shape, vs.shape = ",lambdas.shape, vs.shape)
            log_dict = {}
            for i in range(len(lambdas)):    # send eigenvalues to wandb for tracking
                log_dict[f"lambda{i:02d}"] = lambdas[i]
            wandb.log(log_dict)
    
    wandb.finish()

if __name__ == '__main__':
    main()
