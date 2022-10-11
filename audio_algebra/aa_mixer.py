# AUTOGENERATED! DO NOT EDIT! File to edit: ../aa-mixer.ipynb.

# %% auto 0
__all__ = ['get_alphas_sigmas', 'get_crash_schedule', 'alpha_sigma_to_t', 'sample', 'DiffusionDVAE', 'EmbedBlock', 'AudioAlgebra',
           'aa_demo', 'get_stems_faders', 'main']

# %% ../aa-mixer.ipynb 4
from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import math
import json
import subprocess
import os, sys

import accelerate
import torch
import torchaudio
from torch import optim, nn, Tensor
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.utils import data as torchdata
from tqdm import tqdm, trange
from einops import rearrange, repeat

import wandb

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.hpc import load, save, HostPrinter, freeze
from aeiou.datasets import AudioDataset

# audio-diffusion imports
import pytorch_lightning as pl
from diffusion.pqmf import CachedPQMF as PQMF
from encoders.encoders import AttnResEncoder1D
from autoencoders.soundstream import SoundStreamXLEncoder
from dvae.residual_memcodes import ResidualMemcodes
from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update

# %% ../aa-mixer.ipynb 6
#audio-diffusion stuff 
# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, logits):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    t = get_crash_schedule(t)
    
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], logits).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred



class DiffusionDVAE(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        capacity = 32

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [4, 4, 2, 2, 2]

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )
        self.encoder_ema = deepcopy(self.encoder)

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=2, 
            cond_dim = global_args.latent_dim, 
            pqmf_bands = global_args.pqmf_bands, 
            n_attn_layers=4, 
            c_mults=[256, 256]+[512]*12
        )

        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        self.num_quantizers = global_args.num_quantizers
        if self.num_quantizers > 0:
            quantizer_class = ResidualMemcodes if global_args.num_quantizers > 1 else Memcodes
            
            quantizer_kwargs = {}
            if global_args.num_quantizers > 1:
                quantizer_kwargs["num_quantizers"] = global_args.num_quantizers

            self.quantizer = quantizer_class(
                dim=global_args.latent_dim,
                heads=global_args.num_heads,
                num_codes=global_args.codebook_size,
                temperature=1.,
                **quantizer_kwargs
            )

            self.quantizer_ema = deepcopy(self.quantizer)
            
        self.demo_reals_shape = None #overwrite thie later

    def encode(self, *args, **kwargs):
        if self.training:
            return self.encoder(*args, **kwargs)
        return self.encoder_ema(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.training:
            return self.diffusion(*args, **kwargs)
        return self.diffusion_ema(*args, **kwargs)
    
    def encode_it(self, demo_reals):
        encoder_input = demo_reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(demo_reals)

        encoder_input = encoder_input.to(self.device)
        self.demo_reals_shape = demo_reals.shape
        
        # noise is only used for decoding tbh!
        #noise = torch.randn([demo_reals.shape[0], 2, self.demo_samples]).to(self.device)

        with torch.no_grad():
            embeddings = self.encoder_ema(encoder_input)
            if self.quantized:
                embeddings = rearrange(embeddings, 'b d n -> b n d') # Rearrange for Memcodes
                embeddings, _= self.quantizer_ema(embeddings)
                embeddings = rearrange(embeddings, 'b n d -> b d n')

            embeddings = torch.tanh(embeddings)
            return embeddings#, noise
        
    def decode_it(self, embeddings, demo_batch_size=None, demo_steps=35):
        if None==demo_batch_size: demo_batch_size = self.demo_reals_shape[0]
        noise = torch.randn([self.demo_reals_shape[0], 2, self.demo_samples]).to(self.device)
        fake_batches = sample(self.diffusion_ema, noise, demo_steps, 0, embeddings)
        audio_out = rearrange(fake_batches, 'b d n -> d (b n)') # Put the demos together
        return audio_out

# %% ../aa-mixer.ipynb 12
class EmbedBlock(nn.Module):
    def __init__(self, dims:int, **kwargs) -> None:
        super().__init__()
        self.lin = nn.Linear(dims, dims, **kwargs)
        #self.act = nn.LeakyReLU()
        self.act = F.relu # Mish()
        self.bn = nn.BatchNorm1d(dims)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        #x = rearrange(x, 'b d n -> b n d') # gotta rearrange for bn
        #x = self.bn(x)
        #x = rearrange(x, 'b n d -> b d n') # and undo rearrange for later layers
        return self.act(x)


class AudioAlgebra(nn.Module):
    """
    Main AudioAlgebra model
    """
    def __init__(self, global_args, device, enc_model, trivial=True):
        super().__init__()
        self.trivial = trivial      # trivial=True means trivial (i.e. no) re-embedding
        self.device = device
        self.enc_model = enc_model
        self.dims = global_args.latent_dim
        self.sample_size = global_args.sample_size
        self.num_quantizers = global_args.num_quantizers

        self.reembedding = nn.Sequential(  # something simple at first
            #EmbedBlock(self.dims),
            #EmbedBlock(self.dims),
            #EmbedBlock(self.dims),
            #EmbedBlock(self.dims),
            #EmbedBlock(self.dims),
            nn.Linear(self.dims,self.dims)
            )

    def forward(self,
        stems:list,   # list of torch tensors denoting (chunked) solo audio parts to be mixed together
        faders:list   # list of gain values to be applied to each stem
        ):
        """We're going to 'on the fly' mix the stems according to the fader settings and generate
        frozen-encoder embeddings for each (fader-adjusted) stem and for the total mix.
        "z0" denotes an embedding from the frozen encoder, "z" denotes re-mapped embeddings
        in (hopefully) the learned vector space"""
        with torch.cuda.amp.autocast():
            zs, z0s, zsum, z0sum = [], [], None, None
            mix = torch.zeros_like(stems[0]).float()
            for s, f in zip(stems, faders):   # encode a bunch of stems at different fader settings
                mix_s = s * f                 # audio stem adjusted by gain fader f
                with torch.no_grad():
                    z0 = self.enc_model.encode_it(mix_s)  # encode the stem
                z0sum = z0 if z0sum is None else z0sum + z0 
                z0 = rearrange(z0, 'b d n -> b n d')
                #----------------------------------
                z = z0 if self.trivial else self.reembedding(z0).float()   # <-- this is the main work of the model
                #----------------------------------
                zsum = z if zsum is None else zsum + z # compute the sum of all the z's. we'll end up using this in our (metric) loss as "pred"
                mix += mix_s              # save a record of full audio mix
                zs.append(z)              # save a list of individual z's
                z0s.append(z0)            # save a list of individual z0's

            with torch.no_grad():
                z0mix = self.enc_model.encode_it(mix)  # encode the mix
            z0mix = rearrange(z0mix, 'b d n -> b n d')
            zmix = self.reembedding(z0mix).float()        # map that according to our learned re-embedding. this will be the "target" in the metric loss
            z0mix = rearrange(z0mix, 'b n d -> b d n')
            
            archive = {'zs':zs, 'mix':mix, 'znegsum':None, 'z0s': z0s, 'z0sum':z0sum, 'z0mix':z0mix}

        return zsum, zmix, archive    # zsum = pred, zmix = target, and "archive" of extra stuff zs & zmix are just for extra info


    def mag(self, v):
        return torch.norm( v, dim=(1,2) ) # L2 / Frobenius / Euclidean

    def distance(self, pred, targ):
        return self.mag(pred - targ)
    

    def loss(self, zsum, zmix, archive, margin=1.0, loss_type='noshrink'):
        with torch.cuda.amp.autocast():
            dist = self.distance(zsum, zmix) # for each member of batch, compute distance
            loss = (dist**2).mean()  # mean across batch; so loss range doesn't change w/ batch_size hyperparam
            if ('triplet'==loss_type) and (archive['znegsum'] is not None):
                negdist = self.distance(archive['znegsum'], zmix)
                negdist = negdist * (negdist < margin)   # beyond margin, do nothing
                loss = F.relu( (dist**2).mean() - (negdist**2).mean() ) # relu gets us hinge of L2
            if ('noshrink' == loss_type):                       # TODO: THIS DOESN"T HELP try to preserve original magnitudes of of vectors 
                magdiffs2 = [ ( self.mag(z) - self.mag(z0) )**2 for (z,z0) in zip(archive['zs'], archive['z0s']) ]
                loss += 1/300*(sum(magdiffs2)/len(magdiffs2)).mean() # mean of l2 of diff in vector mag  extra .mean() for good measure  
        return loss

# %% ../aa-mixer.ipynb 14
def aa_demo(autoencoder, log_dict, zsum, zmix, step, demo_steps=35, sr=48000):
    "log decoded audio for zsum and zmix"
    for var,name in zip([zsum, zmix],['zsum','zmix']):
        fake_audio = autoencoder.decode_it(var, demo_steps=demo_steps)
        filename = f'{name}_{step:08}.wav'
        fake_audio = fake_audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, fake_audio, self.sample_rate)
        log_dict[name] = wandb.Audio(filename, sample_rate=sr, caption=name)   
        #log_dict[f'{name}_spec'] = wandb.Image( tokens_spectrogram_image(var.detach()) )
    return log_dict

# %% ../aa-mixer.ipynb 16
def get_stems_faders(batch, dl:torchdata.DataLoader, maxstems=6):
    "grab some more audio stems and set fader values"
    nstems = 1 + int(torch.randint(maxstems-1,(1,1))[0][0].numpy()) # an int between 1 and maxstems, PyTorch style :-/
    #print("nstems =",nstems)
    faders = 2*torch.rand(nstems)-1  # fader gains can be from -1 to 1
    stems = [batch]
    dl_iter = iter(dl)
    for i in range(nstems-1):
        stems.append(next(dl_iter)) 
    return stems, faders

# %% ../aa-mixer.ipynb 26
def main():

    args = get_all_args()
    torch.manual_seed(args.seed)

    try:
        mp.set_start_method(args.start_method)
    except RuntimeError:
        pass

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    hprint = HostPrinter(accelerator)
    hprint(f'Using device: {device}')

    encoder_choices = ['ad','icebox']
    encoder_choice = encoder_choices[0]
    hprint(f"Using {encoder_choice} as encoder")
    if 'icebox' == encoder_choice:
        args.latent_dim = 64  # overwrite latent_dim with what Jukebox requires
        encoder = IceBoxModel(args, device)
    elif 'ad' == encoder_choice:
        dvae = DiffusionDVAE(args, device)
        #dvae = setup_weights(dvae, accelerator, device)
        #encoder = dvae.encoder
        #freeze(dvae)

    hprint("Setting up AA model")
    aa_model = AudioAlgebra(args, device, dvae)

    hprint(f'  AA Model Parameters: {n_params(aa_model)}')

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args)
        config['params'] = n_params(aa_model)
        wandb.init(project=args.name, config=config, save_code=True)

    opt = optim.Adam([*aa_model.reembedding.parameters()], lr=4e-5)

    hprint("Setting up dataset")
    train_set = MultiStemDataset([args.training_dir], args)
    train_dl = torchdata.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    hprint("Calling accelerator.prepare")
    aa_model, opt, train_dl, dvae = accelerator.prepare(aa_model, opt, train_dl, dvae)

    hprint("Setting up frozen encoder model weights")
    dvae = setup_weights(dvae, accelerator)
    freeze(accelerator.unwrap_model(dvae))
    #encoder = dvae.encoder 

    hprint("Setting up wandb")
    if use_wandb:
        wandb.watch(aa_model)

    hprint("Checking for checkpoint")
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        accelerator.unwrap_model(aa_model).load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        del ckpt
    else:
        epoch = 0
        step = 0

    # all set up, let's go
    hprint("Let's go...")
    try:
        while True:  # training loop
            #print(f"Starting epoch {epoch}")
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                batch = batch[0]       # first elem is the audio, 2nd is the filename which we don't need
                #hprint(f"e{epoch} s{step}: got batch. batch.shape = {batch.shape}")
                opt.zero_grad()

                # "batch" is actually not going to have all the data we want. We could rewrite the dataloader to fix this,
                # but instead I just added get_stems_faders() which grabs "even more" audio to go with "batch"
                stems, faders = get_stems_faders(batch, train_dl)

                zsum, zmix, zarchive = accelerator.unwrap_model(aa_model).forward(stems,faders)  # Here's the model's .forward
                loss = accelerator.unwrap_model(aa_model).loss(zsum, zmix, zarchive)
                accelerator.backward(loss)
                opt.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                    if use_wandb:
                        log_dict = {
                            'epoch': epoch,
                            'loss': loss.item(),
                            #'lr': sched.get_last_lr()[0],
                            'zsum_pca': pca_point_cloud(zsum.detach()),
                            'zmix_pca': pca_point_cloud(zmix.detach())
                        }

                        if (step % args.demo_every == 0):                                                    
                            hprint("\nMaking demo stuff")

                            mix_filename = f'mix_{step:08}.wav'
                            reals = zarchive['mix'].clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                            reals = rearrange(reals, 'b d n -> d (b n)')
                            print("reals.shape = ",reals.shape)
                            torchaudio.save(mix_filename, reals, args.sample_rate)
                            log_dict['mix'] = wandb.Audio(mix_filename, sample_rate=args.sample_rate, caption='mix')

                            #demo(accelerator.unwrap_model(dvae), log_dict, zsum.detach(), zmix.detach(),  batch.shape[-1], step)
                            zsum = zarchive['z0sum'].detach() # rearrange(zarchive['z0sum'], 'b n d -> b d n').detach()
                            zmix = zarchive['z0mix'].detach() #rearrange(zarchive['z0mix'], 'b n d -> b d n').detach()

                            hprint(f"zsum.shape = {zsum.shape}")
                            noise = torch.randn([zsum.shape[0], 2, batch.shape[-1]]).to(accelerator.device)
                            accelerator.unwrap_model(dvae).diffusion_ema.to(accelerator.device)
                            model_fn = make_cond_model_fn(accelerator.unwrap_model(dvae).diffusion_ema, zsum)
                            hprint(f"noise.shape = {noise.shape}")

                            # Run the sampler
                            with torch.cuda.amp.autocast():
                                hprint("Calling sampler for zsum")
                                fakes = sample(accelerator.unwrap_model(dvae).diffusion_ema, noise, args.demo_steps, 1, zsum)
                            fakes = rearrange(fakes, 'b d n -> d (b n)')
                            zsum_filename = f'zsum_{step:08}.wav'
                            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                            torchaudio.save(zsum_filename, fakes, args.sample_rate)
                            log_dict['zsum'] = wandb.Audio(zsum_filename, sample_rate=args.sample_rate, caption='zsum')
                            
                            with torch.cuda.amp.autocast():
                                hprint("Calling sampler for zmix")
                                fakes = sample(accelerator.unwrap_model(dvae).diffusion_ema, noise, args.demo_steps, 1, zmix)
                            fakes = rearrange(fakes, 'b d n -> d (b n)')
                            zmix_filename = f'zmix_{step:08}.wav'
                            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                            torchaudio.save(zmix_filename, fakes, args.sample_rate)
                            log_dict['zmix'] = wandb.Audio(zmix_filename, sample_rate=args.sample_rate, caption='zmix')
                            hprint("Done making demo stuff")
                            
                    if use_wandb: wandb.log(log_dict, step=step)

                if step > 0 and step % args.checkpoint_every == 0:
                    save(accelerator, args, aa_model, opt, epoch, step)

                step += 1
            epoch += 1
    except RuntimeError as err:  # ??
        import requests
        import datetime
        ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        hprint(f'ERROR at {ts} on {resp.text} {device}: {type(err).__name__}: {err}', flush=True)
        raise err
    except KeyboardInterrupt:
        pass

# %% ../aa-mixer.ipynb 27
# Not needed if listed in console_scripts in settings.ini
if __name__ == '__main__' and "get_ipython" not in dir():  # don't execute in notebook
    main() 
