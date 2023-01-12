#!/usr/bin/env python3

# Author: Scott H. Hawley 

from prefigure.prefigure import get_all_args, push_wandb_config
import math
import json
import subprocess
import os, sys
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchaudio
from torch import optim, nn, utils, Tensor
from torch.nn import functional as F

from tqdm.auto import tqdm, trange
from einops import rearrange, repeat

import wandb

from aeiou.viz import *
#from aeiou.hpc import load, save, HostPrinter, freeze
from aeiou.datasets import AudioDataset

# audio-diffusion imports
from diffusion.pqmf import CachedPQMF as PQMF
from encoders.encoders import AttnResEncoder1D
from autoencoders.soundstream import SoundStreamXLEncoder
from dvae.residual_memcodes import ResidualMemcodes
from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update

from audio_algebra.aa_mixer import * 

# Lightning: 1. import Lightning
import pytorch_lightning as pl


# Lightning: 2. define LightningModule
class AAMixerModule(pl.LightningModule):
    def __init__(self):
        super().__init__(given_model, aa_model, train_dl)
        self.given_model, self.aa_model, self.train_dl = given_model, aa_model, train_dl
        self.train_iter = iter(train_dl)

    def training_step(self, batch, batch_idx):   

        stems, faders, train_iter = get_stems_faders(batch, self.train_iter, self.train_dl)

        # vicreg: 1. invariance
        zsum, zmix, archive = do_mixing(stems, faders, self.given_model, self.aa_model, module.device, debug=debug)
        mix_loss = mseloss(zsum, zmix)  

        var_loss = (vicreg_var_loss(zsum) + vicreg_var_loss(zmix))/2    # vicreg: 2. variance
        cov_loss = (vicreg_cov_loss(zsum) + vicreg_cov_loss(zmix))/2    # vicreg: 3. covariance

        # reconstruction loss: inversion of aa map h^{-1}(z): z -> y,  i.e. train the aa decoder
        y = self.given_model.encode(batch)
        z, yrecon = self.aa_model.forward(y)   # (re)compute ys for one batch (not stems&faders)
        aa_recon_loss = mseloss(y, yrecon)     
        aa_recon_loss = aa_recon_loss + mseloss(archive['ymix'], archive['ymix_recon'])  # also recon of the  mix ecoding
        #aa_recon_loss  = aa_recon_loss +  mseloss(archive['ysum'], archive['yrecon_sum']) # Never use this:  ysum shouldn't matter / is poorly defined
    
        loss = mix_loss + var_loss + cov_loss + aa_recon_loss     # --- full loss function

        log_dict = {}
        log_dict['loss'] = loss.detach()                    # --- this is the full loss 
        log_dict['mix_loss'] = mix_loss.detach() 
        log_dict['aa_recon_loss'] = aa_recon_loss.detach()
        log_dict['var_loss'] = var_loss.detach() 
        log_dict['cov_loss'] = cov_loss.detach() 
        log_dict['learning_rate'] = opt.param_groups[0]['lr']
        log_dict['epoch'] = epoch
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def configure_optimizers(self):
        opt =  optim.Adam([*self.aa_model.parameters()], lr=5e-4)  # Adam optimizer
        return opt

    def configure_scheduler(self, opt, max_lr=0.001, total_steps=80000):
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)


### MAIN ### 
def main(): 
    args = get_all_args()

    device = get_device()
    print('Using device:', device)
    torch.manual_seed(args.seed)


    # Lightning: 3. define a dataset
    print("Setting up dataset")
    torch.manual_seed(args.seed)
    train_set = AudioDataset([args.training_dir], load_frac=args.load_frac) 
    train_dl = utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                    num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # TODO: make val unique! for now just repeat train & hope for no repeats (train is shuffled, val is not)
    val_set = AudioDataset([args.training_dir], load_frac=args.load_frac/4)
    val_dl = utils.data.DataLoader(train_set, args.batch_size, shuffle=False,
                    num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    torch.manual_seed(args.seed)  # one more seed init for ordering of iterator
    val_iter, train_iter = iter(val_dl), iter(train_dl)
    print("Dataset ready to go! ")



    # Finishing up Lighting 2: define the Lightning module
    # init the given autoencoder
    given_model = DiffusionDVAE.load_from_checkpoint(ckpt_file, global_args=global_args)
    given_model.eval() # disable randomness, dropout, etc...
    given_model.demo_samples, given_model.quantized  = args.sample_size,   args.num_quantizers > 0
    freeze(given_model)  # freeze the weights for inference
    print("Given Autoencoder is ready to go!")

    # init the aa model
    aa_model = AudioAlgebra() # use defaults

    aa_mixer = AAMixerModule(given_model, aa_model, train_dl) # the lightning module
    print("aa_mixer LightningModule ready to go!")




    # Lightning: 4: Train the model  --- add more options
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    wandb_logger.watch(aa_mixer)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=40,
    )
    trainer.fit(model=aa_mixer, train_dataloaders=train_dl, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()