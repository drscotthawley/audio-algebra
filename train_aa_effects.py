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
from aeiou.hpc import freeze
from audio_algebra.datasets import DualEffectsDataset
from audio_algebra.aa_effects import do_mixing, AudioAlgebra
from audio_algebra.given_models import SpectrogramAE, MagSpectrogramAE, MelSpectrogramAE, DVAEWrapper
from audio_algebra.DiffusionDVAE import DiffusionDVAE, sample


# Lightning: 1. import Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

# Lightning: 2. define LightningModule
class AAEffectsModule(pl.LightningModule):
    def __init__(self, given_model, aa_model, train_dl):
        super().__init__()
        self.given_model, self.aa_model, self.train_dl = given_model, aa_model, train_dl
        self.train_iter = iter(train_dl)
        self.batch_shape = None 

    def training_step(self, batch, batch_idx):   
        
        # vicreg: 1. invariance
        archive = do_mixing(batch, self.given_model, self.aa_model, self.device)
        # zs are the projections from aa_model,  archive["yz"] are reps from given_model
        za1, zb1, za2, zb2 = archive["zs"] # a & b are two audio clips, 1 and 2 are effects
        za2_guess = zb2 - zb1 + za1     # try to enforce enfoce algebraic property 
        zb2_guess = za2 - za1 + zb1
        mix_loss = (mseloss(za2_guess, za2)  +  mseloss(zb2_guess, zb2))/2
        
        var_loss = (vicreg_var_loss(za2_guess) + vicreg_var_loss(zb2_guess))/2    # vicreg: 2. variance
        cov_loss = (vicreg_cov_loss(za2_guess) + vicreg_cov_loss(zb2_guess))/2    # vicreg: 3. covariance

        # reconstruction loss: inversion of aa map h^{-1}(z): z -> y,  i.e. train the aa decoder
        aa_recon_loss = mseloss(archive["yrecons"][0], archive["ys"][0])
        for i in range(1,4):
            aa_recon_loss += mseloss(archive["yrecons"][i], archive["ys"][i]) 

        loss = mix_loss + var_loss + cov_loss + aa_recon_loss     # --- full loss function

        # logging during training
        log_dict = {}
        #log_dict['loss'] = loss.detach()                    # --- this is the full loss 
        log_dict['mix_loss'] = mix_loss.detach() 
        log_dict['aa_recon_loss'] = aa_recon_loss.detach()
        log_dict['var_loss'] = var_loss.detach() 
        log_dict['cov_loss'] = cov_loss.detach() 
        #log_dict['learning_rate'] = self.opt.param_groups[0]['lr']
        
        
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer =  optim.Adam([*self.aa_model.parameters()], lr=5e-4)  # Adam optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, val_dl, given_model, aa_model, device, global_args):
        super().__init__()
        self.given_model, self.aa_model, self.device = given_model, aa_model, device
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.demo_dl = iter(val_dl)
        self.sample_rate = global_args.sample_rate

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        last_demo_step = trainer.global_step

        batch = next(self.demo_dl)
        archive = do_mixing(batch, self.given_model, self.aa_model, self.device)
        # zs are the projections from aa_model, archive["yz"] are reps from given_model
        za1, zb1, za2, zb2 = archive["zs"] # a & b are two audio clips, 1 and 2 are effects
        za2_guess = zb2 - zb1 + za1     # try to enforce enfoce algebraic property 
        zb2_guess = za2 - za1 + zb1

        #ya1, yb1, ya2, yb2 = archive["ys"] # a & b are two audio clips, 1 and 2 are effects

        try:   # don't crash the whole run just because logging fails
            log_dict = {}
            e1names, e2names = batch["e1"], batch["e2"]  # these are batches of names of audio effects 

            for var, name in zip([za1, zb1, za2, zb2],["za1", "zb1", "za2", "zb2"]):
               #log_dict[f'emb_table_{name}'] = embeddings_table(var)
               var = rearrange(var,'b d n -> d (b n)')   # pack batches as successive groups of time-domain samples
               log_dict[f'{name}_3dpca'] = pca_point_cloud(var, output_type='plotly', mode='lines+markers')
               log_dict[f'{name}_spec'] = wandb.Image(tokens_spectrogram_image(var))

            for key in ["a","b", "a1","b1", "a2","b2"]:  # audio
                audio = batch[key]
                filename = f'{key}_{trainer.global_step:08}.wav'
                audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, audio, self.sample_rate)
                log_dict[f'{key}'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Inputs')

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)







### MAIN ### 
def main(): 
    args = get_all_args()

    device = torch.device('cuda')  # this code runs on clusters only
    print('Using device:', device)
    torch.manual_seed(args.seed)


    # Lightning: 3. define a dataset
    print("Setting up dataset")
    torch.manual_seed(args.seed)
    train_set = DualEffectsDataset([args.training_dir], load_frac=args.load_frac, debug=False) 
    train_dl = utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                    num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # TODO: make val unique! for now just repeat train & hope for no repeats (train is shuffled, val is not)
    val_set = DualEffectsDataset([args.training_dir], load_frac=args.load_frac/4, debug=False)
    val_dl = utils.data.DataLoader(train_set, args.batch_size, shuffle=False,
                    num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    torch.manual_seed(args.seed)  # one more seed init for ordering of iterator
    val_iter, train_iter = iter(val_dl), iter(train_dl)
    print("Dataset ready to go! ")

    # Finishing up Lighting, 2: define the Lightning module
    # init the given autoencoder
    #given_model = SpectrogramAE()
    given_model = DiffusionDVAE.load_from_checkpoint(args.dvae_ckpt_file, global_args=args)
    #given_model = DVAEWrapper.load_from_checkpoint(args.dvae_ckpt_file, global_args=args)
    given_model.demo_samples, given_model.quantized  = args.sample_size,   args.num_quantizers > 0
    given_model.eval() # disable randomness, dropout, etc...
    freeze(given_model)  # freeze the weights for inference
    print("Given Autoencoder is ready to go!")

    # init the aa model
    aa_use_bn = False  # batch norm?
    aa_use_resid = True # use residual connections? (doesn't make much difference tbh)
    emb_dims = args.latent_dim # input size to aa model
    hidden_dims = emb_dims   # number of hidden dimensions in aa model. usually was 64
    trivial = False  # aa_model is a no-op when this is true
    debug = True
    aa_model = AudioAlgebra(dims=emb_dims, hidden_dims=hidden_dims, use_bn=aa_use_bn, resid=aa_use_resid, trivial=trivial).to(device)

    aa_effects = AAEffectsModule(given_model, aa_model, train_dl) # the lightning module
    print("aa_effects LightningModule ready to go!")

    # Lightning: 4: Train the model  --- add more options
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(val_dl, given_model, aa_model, device, global_args)
    exc_callback = ExceptionCallback()

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(aa_effects)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        #strategy='ddp',
        strategy="ddp_find_unused_parameters_false",
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, lr_monitor, demo_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=40,
    )
    trainer.fit(model=aa_effects, train_dataloaders=train_dl, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
