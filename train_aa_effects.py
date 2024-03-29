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

from pathlib import Path

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
from audio_algebra.aa_effects import *
from audio_algebra.given_models import SpectrogramAE, MagSpectrogramAE, MelSpectrogramAE, DVAEWrapper
from audio_algebra.DiffusionDVAE import DiffusionDVAE, sample


from audiomentations import *   # list of effects
 

# Lightning: 1. import Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

def vicreg_var_loss_l2(z, gamma=1, eps=1e-4):
    std_z = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu( (gamma - std_z))**2 )   # the relu gets us the max(0, ...)

vicreg_var_loss = vicreg_var_loss_l2


# Lightning: 2. define LightningModule
class AAEffectsModule(pl.LightningModule):
    def __init__(self, given_model, aa_model, train_dl, debug=False):
        super().__init__()
        self.given_model, self.aa_model, self.train_dl, self.debug = given_model, aa_model, train_dl, debug
        self.train_iter = iter(train_dl)
        self.batch_shape = None 
        

    def training_step(self, batch, batch_idx):   
        if self.debug: print("\nStarting AAEffectsModule.training_step")

        mseloss = nn.MSELoss()
        
        # vicreg: 1. invariance
        if self.debug: print("   calling do_mixing")
        with torch.cuda.amp.autocast():
            archive = do_mixing(batch, self.given_model, self.aa_model, self.device)
            # zs are the projections from aa_model,  archive["yz"] are reps from given_model
            [za1, zb1, za2, zb2] = [x.float() for x in archive["zs"]] # a & b are two audio clips, 1 and 2 are effects
            
            za2_guess = zb2 - zb1 + za1     # try to enforce enfoce algebraic property 
            zb2_guess = za2 - za1 + zb1
            mix_loss = (mseloss(za2_guess, za2)  +  mseloss(zb2_guess, zb2))/2
        
            var_loss = (vicreg_var_loss(za1) + vicreg_var_loss(za2) + vicreg_var_loss(zb1) + vicreg_var_loss(zb2))/4  # vicreg: 2. variance
            cov_loss = (vicreg_cov_loss(za1) + vicreg_cov_loss(za2) + vicreg_cov_loss(zb1) + vicreg_cov_loss(zb2))/4  # vicreg: 3. covariance

            # reconstruction loss: inversion of aa map h^{-1}(z): z -> y,  i.e. train the aa decoder
            aa_recon_loss = mseloss(archive["yrecons"][0].float(), archive["ys"][0].float())
            for i in range(1,4):
                aa_recon_loss += mseloss(archive["yrecons"][i].float(), archive["ys"][i].float()) 

            loss = mix_loss + var_loss + cov_loss + aa_recon_loss     # --- full loss function

        if self.debug: print("   full loss calculated. setting log_dict...")
        # logging during training
        log_dict = {
            'tloss': loss.detach(),
            'mix_loss': mix_loss.detach(),
            'var_loss': var_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'aa_recon_loss': aa_recon_loss.detach()
        }
        #log_dict['learning_rate'] = self.opt.param_groups[0]['lr']
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        if self.debug: print("   training_step: returning loss\n")

        return loss

    def configure_optimizers(self):
        optimizer =  optim.Adam([*self.aa_model.parameters()], lr=5e-4)  # Adam optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, val_dl, given_model, aa_model, device, global_args, wandb_logger):
        super().__init__()
        self.given_model, self.aa_model, self.device = given_model, aa_model, device
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.demo_dl = iter(val_dl)
        self.sample_rate = global_args.sample_rate
        self.debug = True
        self.wandb_logger = wandb_logger

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        if self.debug: print("\nin DemoCallback.on_train_batch_end")
        last_demo_step = trainer.global_step

        batch = next(self.demo_dl)
        with torch.no_grad():
            archive = do_mixing(batch, self.given_model, self.aa_model, self.device)
        # zs are the projections from aa_model, archive["yz"] are reps from given_model
        za1, zb1, za2, zb2 = archive["zs"] # a & b are two audio clips, 1 and 2 are effects
        za2_guess = zb2 - zb1 + za1     # try to enforce enfoce algebraic property 
        zb2_guess = za2 - za1 + zb1

        if self.debug: print("trying to log to wandb")
        #try:   # don't crash the whole run just because logging fails
        if True:
            log_dict = {}
            e1names, e2names = batch["e1"], batch["e2"]  # these are batches of names of audio effects 
            if self.debug: print("effects: [1,2]: ",list(zip(e1names, e2names)))

            audios, melspecs, tokenspecs, pcs = [],[],[],[]
            for var, name in zip([za1, za2, zb1, zb2],["za1", "za2", "zb1", "zb2"]):
                if self.debug: print(" logging: name =",name, ", var.shape =",var.shape) 
                log_dict[f'{name}_embeddings'] = embeddings_table(var)
                log_dict[f'{name}_3dpca'] = pca_point_cloud(var, output_type='wandbobj', mode='lines+markers')
                log_dict[f'{name}_spec'] = wandb.Image(tokens_spectrogram_image(var))
                pcs.append(log_dict[f'{name}_3dpca'])
                tokenspecs.append(log_dict[f'{name}_spec'])

            for key in ["a1","a2", "b1","b2"]:  # audio inputs a & b, with effects 1 and 2 applied
                if self.debug: print("Logging: key =",key)
                audio = batch[key]
                audio = rearrange(audio,'b d n -> d (b n)')   # pack batches as successive groups of time-domain samples
                if self.debug: print("    new audio.shape = ",audio.shape)
                log_dict[f'{key}_melspec_left'] = wandb.Image(audio_spectrogram_image(audio))
                filename = f'{key}_{trainer.global_step:08}.wav'
                audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, audio, self.sample_rate)
                log_dict[f'{key}_audio'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Inputs')
                audios.append(log_dict[f'{key}_audio'])
                melspecs.append(log_dict[f'{key}_melspec_left'])

            columns = ['soundfile', 'melspec', 'tokenspec', '3dpca']
            data = [[s,m,t,p] for s, m, t, p in zip(audios, melspecs, tokenspecs,pcs)]
            log_dict["global_step"] = trainer.global_step
            trainer.logger.experiment.log(log_dict)

            #columns = ['soundfile', 'ground truth', 'prediction']
            #grount_truth = [ 'yes' for y_true in range(50)]
            #predicted = ['yes' if y_pred%2 ==0 else 'no' for y_pred in range(50)]
            #audio_paths = filter(lambda x: 'wav' in x.name, Path('./').iterdir())
            #n = 10
            #data = [[wandb.Audio(str(x_sound)), y_true, y_pred] for x_sound, y_true, y_pred in zip(audio_paths, grount_truth[:n], predicted[:n])]
            #self.wandb_logger.log_table(key='sample_table', columns=columns, data=data, step=trainer.global_step)
            #trainer.logger.log_table(key='sample_table', columns=columns, data=data)
            if self.debug: print("trainer logger set")

        #except Exception as e:
        #    print(f'{type(e).__name__}: {e}', file=sys.stderr)


### MAIN ### 
def main(): 
    args = get_all_args()
    print("args = ",args)

    device = torch.device('cuda')  # this code runs on clusters only
    print('Using device:', device)
    torch.manual_seed(args.seed)


    # Lightning: 3. define a dataset
    print("Setting up dataset")
    torch.manual_seed(args.seed)
    effects_list = [Gain, BandPassFilter, BandStopFilter, HighPassFilter, LowPassFilter]

    train_set = DualEffectsDataset([args.training_dir], load_frac=args.load_frac, effects_list=effects_list) 
    train_dl = utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                    num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # TODO: make val unique! for now just repeat train & hope for no repeats (train is shuffled, val is not)
    val_set = DualEffectsDataset([args.training_dir], load_frac=args.load_frac/4, effects_list=effects_list)
    val_dl = utils.data.DataLoader(train_set, args.num_demos, shuffle=False,
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
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(aa_effects)
    push_wandb_config(wandb_logger, args)

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(val_dl, given_model, aa_model, device, args, wandb_logger)
    exc_callback = ExceptionCallback()

    lr_monitor = LearningRateMonitor(logging_interval='step')

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
        max_epochs=40000,
    )
    trainer.fit(model=aa_effects, train_dataloaders=train_dl, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
