#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A CLI or GUI-based demo for processing audio using embeddings. Like Destructo
GUI uses Gradio
CLI based (so Discord-bot-compatible) but will call up GUI if no filename given

Candidate names:
    Simthesizer / Simthesize / Symth 
    MIRAGE: Music Information Retrieval-based Autoencoder for Generation via Entropy
    RAGE: Retrieval-based Audio Generation via Entropy
    MIRACLE: Music Information Retrieval-based Audio Creation via Lossy Encoding 
    SPECTER: Sound Production via Efficient Compression Techniques and Entropic Reconstruction
    ReGALE: Reinterpreting Generative Audio via Lossy Encoding 

AUTHOR: Scott H. Hawley @drscotthawley 
LICENSE: Creative Commons 4.0, commercial usage allowed. Just cite this gist URL in a comment to be nice. 

NB: This entire source code can be pasted into a single Juptyer notebook cell and it will run as-is, calling the GUI
"""
print("Starting imports...")
import os
import sys 
from pathlib import Path
import argparse 
import numpy as np 
import torch
import torchaudio
import matplotlib.cm as cm    
import gradio as gr
import traceback
import math
from functools import partial
from aeiou.core import load_audio, get_device
print("   Starting CLAPDAE import")
from audio_algebra.given_models import CLAPDAE#, KDiff_CLAPDAE 
#import plotly.express as px 
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
print("Finished with imports.")
from time import sleep
import random
from einops import rearrange
import csv 
from PIL import Image
import PIL
import requests 
from io import BytesIO


# global vars
model, current_model_choice = None, '' # global vars to keep (same) model initialized thru multiple GUI ops 
half_precision = False 
GRADIO_AUDIO_INFO = {'sr':48000, 'mono_in':False, 'tensor_in':False, 'int_in':True, 'channels_first':False} # used in a few places
GRADIO_TAB_STATE = "tab1"
#PLOT_BGCOLOR = 'rgb(33,41,54)'  # panels in dark mode
PLOT_BGCOLOR = 'rgb(12,15,24)'   # margins in dark mode
cached_logo_fig, cached_pixels = None, None


def unpack_audio_tup(audio_tup, verbose=True):
    """utility routine for process_audio: 
    turn gradio's audio object into something like my models expect:
    My routines want stereo, channels-first, 32-bit float PyTorch tensors, but Gradio gives channels-last int16 numpy arrays
    """
    if audio_tup is None: return (None, None) 

    (sr, waveform) = audio_tup
    if verbose: print("unpack_audio: initially waveform.shape, waveform.dtype =",waveform.shape, waveform.dtype)
    mono_in, tensor_in = len(waveform.shape)==1, torch.is_tensor(waveform)
    if not tensor_in: waveform = torch.tensor(waveform)
    int_in = not torch.is_floating_point(waveform)  # dtype used by gradio
    audio_info = {'sr':sr, 'mono_in':mono_in, 'tensor_in':tensor_in, 'int_in':int_in}
    if verbose: print("audio_info = ",audio_info)
    
    if waveform.dtype == torch.int16: # Rescale to float in range [-1, 1] if necessary
        waveform = waveform / 32768.0
    elif waveform.dtype == torch.int32:
        waveform = waveform / 2147483648.0
    elif waveform.dtype != torch.float32 and waveform.dtype != torch.float64:
        raise ValueError("Input waveform must be float32, float64, int16, or int32")
    waveform = waveform.float() # make sure everything is float type regardless
    if mono_in: 
        waveform = waveform.unsqueeze(0) # add channel dimension if needed
        channels_first = None # undefined
    else:
        channels_first = (waveform.shape[0] < waveform.shape[1])
        audio_info['channels_first'] = channels_first
        if not channels_first: waveform = waveform.transpose(0,1) 
    if verbose: print("unpack_audio: finally waveform.shape, waveform.dtype =",waveform.shape, waveform.dtype)
    return waveform, audio_info
  

def repack_audio_tup(new_waveform, audio_info, verbose=True):
    """utility routine for process_audio:    
            puts things back to the way they were passed in
    """
    if audio_info['mono_in']: 
        new_waveform = new_waveform[0] # remove channels dimension / back to mono
    elif not audio_info['channels_first']:
        new_waveform = new_waveform.transpose(0,1)
    if verbose: print("repack_audio: new_waveform.shape =",new_waveform.shape)
    if audio_info['int_in']: 
        new_waveform = new_waveform.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    if not audio_info['tensor_in']: new_waveform = new_waveform.numpy()
    if verbose: print("new_waveform.dtype =", new_waveform.dtype)
    return (audio_info['sr'], new_waveform) 


def half_it(x, debug=True, for_audio_embed=False): 
    "apply half precision if enabled, otherwise a no-op"
    global half_precision
    if (not half_precision) or (x is None): return x
    if x == model: # aren't globals aweome: 
        model.latent_diffusion_model.diffusion_ema.half()
        model.latent_diffae.half()
        model.clap_module.float() # just in case #  don't try to make CLAP work with half
        return model
    elif torch.is_tensor(x): 
        return x.half()
    raise TypeError(f'half_it: unexpected type x = {type(x)}')


def get_model_ready(model_choice, device, verbose=True):
    """utility routine for process_audio: gets model ready if it's not already"""
    global model, current_model_choice, half_precision  # Hate using globals but how else to keep model initialized?

    if verbose: print("current_model_choice, model_choice =", current_model_choice, model_choice)
    half_precision = 'fp16' in model_choice

    if model_choice != current_model_choice:
        # if only difference is precision, don't reload from checkpoint
        if model_choice.replace('-fp16','') == current_model_choice.replace('-fp16',''): 
            if verbose: print("Changing model precision...")
            if 'fp16' in model_choice:
                model = half_it(model)
            else:
                model.float()
        else:
            if verbose: print("Hang on, we need to load a new model...")
            model = CLAPDAE(device=device)
            #model = KDiff_CLAPDAE(device=device) # not working yet
            model.setup()
            model = half_it(model)
        model.eval()
        current_model_choice = model_choice
    elif model is not None:
        if verbose: print(f"Model's already loaded! {model.__class__.__name__}")
    else:
        raise gr.Error("Can't find the right model, sorry.")

        
def lerp(a, b, t):
    """Linear interpolation between two vectors a and b by a factor t"""
    return (1 - t) * a + t * b

def slerp(a, b, t):
    """Spherical interpolation between two vectors a and b by a factor t. Source: thanks ChatGPT"""
    a_norm = a / torch.norm(a, dim=-1, keepdim=True)
    b_norm = b / torch.norm(b, dim=-1, keepdim=True)
    dot = torch.sum(a_norm * b_norm, dim=-1, keepdim=True)
    omega = torch.acos(dot.clamp(-1, 1))
    if omega == 0:
        return a
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega * a_norm
          + torch.sin(      t * omega) / sin_omega * b_norm)

def interp_embeddings(emb1, emb2, interp_scale=0.5, interp_type='Spherical'):
    "interpolation choices. leave it on Spherical since CLAP is normalized"
    if emb2 is None: return emb1
    if interp_type=='Linear':
        return lerp(emb1, emb2, interp_scale)
    else:
        return slerp(emb1, emb2, interp_scale)

    
def sdd_str(x): # shorthand because I print this info a lot when debugging
    if x is not None:
        return f" shape, dtype, device = {x.shape}, {x.dtype}, {x.device}"
    else:
        return "None"

@torch.no_grad()
def crossfade_flatten(ab,     # audio batch
                      cl=0,   # crossfade length in samples, 0=no-op
                      fade_type='sine'): # 'linear'|'sine'|'sqrt'
    "flattens but cross-fades between batch elements. "
    if len(ab.shape)<3: return ab                          # not batched, no need
    if ab.shape[0]==1: return ab[0]                        # only one batch 
    if cl==0: return rearrange( ab, 'b c n -> c (b n)')    # don't crossfade, just flatten

    if not torch.is_tensor(ab): ab = torch.tensor(ab)
    fade_in = torch.linspace(0, 1, steps=cl, dtype=ab.dtype, device=ab.device) 
    if fade_type=='sine': 
        fade_in = torch.sin(fade_in * 0.5*math.pi )
    elif fade_type=='sqrt':
        fade_in = torch.sqrt(fade_in)
    fade_out = fade_in.flip(dims=(0,))
    ab[1:,:,:cl] = ab[1:,:,:cl]*fade_in + ab[:-1,:,-cl:]*fade_out     # apply cross-fades
    chopped_flattened = rearrange( ab[:,:, :-cl], 'b c n -> c (b n)' ) # chop off fade-out regions (and last piece)
    return torch.cat( ( chopped_flattened, ab[-1,:,-cl:]), dim=-1 )    # stick last piece back on 


def preproc_audio_tup_input(audio_tup):
    "sometimes gradio gives you strings instead of tuples, so convert to tuples if needed"
    if audio_tup is not None and isinstance(audio_tup, str):
        if audio_tup=="": return None
        audio_info = GRADIO_AUDIO_INFO
        filename = audio_tup
        waveform = load_audio(filename, sr=audio_info['sr']) 
        audio_tup = repack_audio_tup(waveform, audio_info)   
    return audio_tup

def myPath(x): 
    """none-tolerant path"""
    return Path(x) if x is not None else None
    
    
@torch.no_grad()
def embed_audio_or_text(embedder, waveform=None, text_prompt=""):
    "text prompt supersedes audio"
    if text_prompt: return embedder(text_prompt)
    if waveform is not None: return embedder(waveform) 
    return None





#####---------------- MAIN AUDIO PROCESSING ROUTINE
@torch.no_grad()    
def process_audio(
    audio_tup=None,        # a tuple of (sr, waveform) where sr is sample rate
    text_prompt="",
    audio_tup2=None,  # optional 2nd audio for interpolation
    text_prompt2="",
    interp_scale=0.5, # relative strength of audio_tup & audio_tup2
    init_audio_tup=None, 
    init_strength=0.4,
    cfg_scale=4.0,    # cfg scale, a number
    seed_value=0,          # RNG seed
    batch_size=1,
    #model_choice='CLAPDAE-22s', # a string, the name of the model to use
    crossfade=False,  # crossfade batches or no
    #crossfade_secs=0.0,
    audio_tup_a=None, text_prompt_a="", weight_a=1,   # "king"
    audio_tup_b=None, text_prompt_b="", weight_b=-1,  # "man"
    audio_tup_c=None, text_prompt_c="", weight_c=1,   # "woman"
    verbose=True, # how much info we print while executing
    show_embeddings=False, # whether to show the embeddings plot in the output - only in GUI mode
    model_choice='CLAPDAE-22s', # a string, the name of the model to use
    sample_rate=48000,
    device='cuda',           # pytorch device to use
    ):
    """That code which processes the audio"""
    global half_precision
    
    if verbose:
        print(f"\n\n ----------------------------------------------------------------\n")
        print(f"process_audio: passed in: model_choice = {model_choice},  device = {device}")
        print(f"audio_tup, text_prompt, audio_tup_a, text_prompt_a = ",audio_tup, text_prompt, audio_tup_a, text_prompt_a)
        print(f"interp_scale = {interp_scale}, init_audio_tup = {init_audio_tup},")
    
    
    if not any([audio_tup, text_prompt, audio_tup_a, text_prompt_a]): 
        raise gr.Error("No audio or text prompt passed in.") 
        return None 
    if cfg_scale is None: 
        raise gr.Error("No cfg_scale given.") 
        return None, None

    
    audio_tups = [preproc_audio_tup_input(x) for x in [audio_tup, audio_tup2, audio_tup_a, audio_tup_b, audio_tup_c, init_audio_tup]]
    audio_str, audio2_str, audio_str_a, audio_str_b, audio_str_c, init_audio_str = audio_tups  # nowadays we get file strings, not numpy arrays
                 
    half_precision = 'fp16' in model_choice
    
    try:
        batch_size = int(batch_size)
        if verbose: print("process_audio: calling unpack_audio_tup")
        waveforms, audio_infos = [], []
        for atup in audio_tups:
            w, audinfo = unpack_audio_tup(atup, verbose=verbose)
            waveforms.append(w)
            audio_infos.append(audinfo)
        text_prompts = [text_prompt, text_prompt2, text_prompt_a, text_prompt_b, text_prompt_c]
        wavs2embed = waveforms[:-1]  # don't include init audio when embedding
        assert len(wavs2embed) == len(text_prompts), "Lengths need to match"
        
        get_model_ready(model_choice, device, verbose=verbose) # model might already be loaded but let's be sure
        
        
        ##---------------- Do the actual audio processing ------------------
        for i in range(len(waveforms)):
            if waveforms[i] is not None: waveforms[i] = waveforms[i].to(model.device)  
            
        if verbose: print(f"  ------ Getting CLAP Embeddings:")
        embeddings_list = []
        for waveform, text_prompt in list(zip(wavs2embed, text_prompts)):
            emb = embed_audio_or_text(model.embed, waveform, text_prompt) # TODO: can we batch these? 
            embeddings_list.append(emb)       

            
        # Now manipulate the embeddings!
        if (None not in embeddings_list[0:2]) and (GRADIO_TAB_STATE == "tab1"):  # TODO need to check on Tab state
            print("      --- Interpolating CLAP Embeddings")
            embeddings = interp_embeddings(embeddings_list[0], embeddings_list[1], interp_scale, interp_type='Spherical')  
        elif any(map(lambda item: item is not None, embeddings_list[2:])) and (GRADIO_TAB_STATE == "tab2"):  
            print("      --- AUDIO ALLLLGEEEBRRAA!!!")
            embeddings = torch.zeros([1,1,512]).to(model.device)
            for weight, emb in list(zip( (weight_a, weight_b, weight_c), embeddings_list[2:])):
                if emb is not None: 
                    embeddings += weight * emb
            embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)    # normalize the result
        else:
            raise gr.Error("Don't have sufficient non-None inputs, and/or tab error: GRADIO_TAB_STATE =",GRADIO_TAB_STATE)                     
        embeddings  = half_it(embeddings) # note FP16 decoding sounds bad and clippy, but code's in place anyway

        
        # Init audio basis
        init_audio_latents, init_waveform = None, waveforms[-1]
        print("        init_waveform.shape = ",(None if init_waveform is None else init_waveform.shape))
        if init_waveform is not None:  # init audio
            print(f"      ---- Preparing latents of init_audio")
            init_waveform = half_it(init_waveform)
            # TODO: the following 4 lines are a hack to avoid off-by-1 size mismatches in Flavio's AE.
            new_init_waveform = torch.zeros([2,model.sample_size], device=device, dtype=init_waveform.dtype) 
            minlen = min( init_waveform.shape[-1], new_init_waveform.shape[-1] ) 
            new_init_waveform[:,:minlen] = init_waveform[:,:minlen]
            init_waveform = new_init_waveform
            init_audio_latents = model.latent_diffusion_model.encode(init_waveform.to(device))
            print(f"           init_audio_latents {sdd_str(init_audio_latents)}")


            
        #############  GENERATE NEW AUDIO  ###############
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        cfg_scales = [cfg_scale] # TODO: only allow 1 cfg for no, plan to add more            
        if verbose: print(f"\n  ------ Generating Audio, cfg_scales = {cfg_scales}, batch_size = {batch_size}, init_strength = {init_strength}")
        fakes, fake_latents = model.generate(embeddings, cfg_scales=cfg_scales, 
                                             init_audio_latents=init_audio_latents, init_strength=init_strength, 
                                             batch_size=batch_size, flatten=False)
        crossfade_len = int(crossfade)*int(1.5 * 48000) # we'll do 1.5 seconds  ... :shrug:
        fakes = crossfade_flatten(fakes, cl=crossfade_len, fade_type='sine')

        if verbose: print("\n  ------ Finished generating.  fakes ",sdd_str(fakes))
        ##----------------------Finished with true audio processing--------------------------
        # now get it ready to be returned to gradio or wherever inputs came from 
        
        #if audio_info is None:  # supply gradio defaults if generated from text prompt
        #    audio_info = GRADIO_AUDIO_INFO
        #new_audio_tup = repack_audio_tup(fakes, audio_info, verbose=verbose)
        
        # or what if we just supplied a filename?
        audio_out_filename = "FILENAME_ERROR_BUT_DONT_CARE.wav"
        try: 
            for a, t in list(zip( audio_tups[:-1], text_prompts ) ):
                audio_out_filename += f"{t if t else ('' if a is None else Path(a).name )}_" 
            prompt1 = text_prompt if text_prompt else ( Path(audio_str).name if audio_str is not None else None )
            prompt2 = text_prompt2 if text_prompt2 else ( Path(audio2_str).name if audio2_str is not None else None )
            init_report = 'a-a' # if init_audio_str is None else ( Path(init_audio_str).name if init_audio_str is not None else None)
            audio_out_filename = f"{prompt1}--{interp_scale}--{prompt2}__{init_report}_{init_strength}___{cfg_scale}_{seed_value}.wav".replace(' ','_')
        except Exception as e:
            print(f"Ok, some kind of exception creating the filename: {e}.")
        if verbose: print(f"\n   >>>>>>  Saving to output file {audio_out_filename}, fakes.shape = {fakes.shape}\n")
        
        fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(audio_out_filename, fakes, sample_rate)
        
        new_audio_tup = audio_out_filename # TODO: refactor onece this works
        
        if show_embeddings:
            if verbose: print("Plotting latents graph")
            fig = pca_point_cloud(fake_latents.float(), output_type='plotly', mode='lines+markers', color_scheme='', colormap=cm.Blues, darkmode=PLOT_BGCOLOR)
        else:
            fig = None

    except Exception as e:   
        msg = f"Error processing audio: {e}\n"+traceback.format_exc()
        raise gr.Error(msg) # either shows in GUI or gets printed later in CLI
    if show_embeddings:
        return (new_audio_tup, fig)
    else:
        return new_audio_tup

    
    

def get_examples(filename="mirage_examples.csv"):
    result = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or row[0].startswith('#'): # Ignore comment lines starting with #
                continue
            row = [int(val) if val.isdigit() else float(val) if val.replace('.', '', 1).isdigit() else None if val == 'None' else val for val in row]
            result.append(row)
    examples = result
    print("examples = \n",examples)
    return examples



# these next two routines are just for fun, to fill in the empty plot space at the bottom

def get_non_transparent_pixels_from_url(image_url):
    "thanks ChatGPT for help with this"
    global cached_pixels
    if cached_pixels is not None: 
        print("got logo pixels already")
        return cached_pixels 
    image_data = requests.get(image_url).content
    rgba = np.array( Image.open(BytesIO(image_data)).convert('RGBA').transpose(PIL.Image.FLIP_LEFT_RIGHT) )  # flipping lr is bc of default view from plotly 
    non_transparent_indices = np.where(rgba[..., 3] != 0)
    if len(non_transparent_indices) == 0:  # hmm, maybe white pixels then?
        non_transparent_indices = np.where(rgba[..., 0:3] == 255)
    cached_pixels = np.array([non_transparent_indices[1], non_transparent_indices[0], 0*non_transparent_indices[0]])
    return cached_pixels


def harmonai_logo_3d_plotly(image_url='https://images.squarespace-cdn.com/content/v1/62b318b33406753bcd6a17a5/ffa25141-8195-4916-8acf-7d5a44f08dfe/Transparent_Harmonai+Logo-02+%281%29.png?format=1500w', 
                     downsample_by=10, colormap=cm.Blues, ):
    "builds a 3d model of the Harmonai logo using the image from the website"
    global cached_logo_fig 
    if cached_logo_fig is not None: 
        print("Already got the 3D logo so we don't need to reload!")
        return cached_logo_fig # save to mae it faster next time

    pixels = get_non_transparent_pixels_from_url(image_url)[:,::downsample_by]
    tokens = torch.tensor(pixels[np.newaxis, :, :]).float()
    tokens[:,2,:] *= -1 #flip it? 
    try:
        layout_dict=dict(  scene=dict(  camera=dict( up=dict(x=0, y=0, z=1), eye=dict(x=.08, y=1.1, z=1), ), ) )     
        fig = pca_point_cloud(tokens, color_scheme='', output_type='plotly', colormap=colormap, darkmode=PLOT_BGCOLOR, layout_dict=layout_dict )
    except:
        print("Note: you won't get the Blue colormap without the bleeding-edge version of aeiou")
        fig = pca_point_cloud(tokens, color_scheme='', output_type='plotly', )
    cached_logo_fig = fig
    return fig



def set_active_tab_info(EventData): 
    global GRADIO_TAB_STATE
    print("EventData = ",EventData)
    GRADIO_TAB_STATE = EventData
    return



def run_gui(device, verbose=True, public=False, model_choices=["model1", "model2", "model3"], model_choice=None, 
            cache_examples=False ,**kwargs):
    """Launches the GUI"""

    css_code = """#aud_comp1 {height: 90px;} #aud_comp2 {height: 90px;} #aud_comp3 {height: 90px;} 
                #aud_comp4 {height: 120px;} #check_comp {height: 90px;}
                #submit {height: 90px;}
                #weight1 { height: 200px; width: 200px; transform: rotate(270deg); }
                """
                # #background: url('file=submit_bg1.png'); background-repeat: no-repeat; background-size: 100% 100%; background-position: center;}
    
    with gr.Blocks(title=kwargs['title'], css=css_code, theme=gr.themes.Base()) as demo:
        gr.HTML(f'<center><a href="https://www.harmonai.org/"><img src="https://images.squarespace-cdn.com/content/v1/62b318b33406753bcd6a17a5/ffa25141-8195-4916-8acf-7d5a44f08dfe/Transparent_Harmonai+Logo-02+%281%29.png?format=1500w" alt="Harmonai logo" width="100" height="100"></a><h1>{kwargs["title"]}</h1>{kwargs["description"]}</center>')        
        
        with gr.Tab("(Re)Gen / Interp") as tab1: 
            with gr.Row():   #     label="Input 1"):
                with gr.Box():
                    with gr.Column():
                        audio_1 = gr.Audio(label="Audio Prompt 1", type="filepath", elem_id="aud_comp1", )
                        text_prompt_1 = gr.Textbox(value="", label="Text Prompt 1", info="Overrides Audio Prompt 1")
                with gr.Box():
                    with gr.Column():   # label="Model Controls"):
                        audio_2 = gr.Audio(label="Audio Prompt 2", type="filepath", elem_id="aud_comp2")
                        #interp_type = gr.Radio(['Spherical', 'Linear'], value='Spherical', label="Interpolation type (Leave it on Spherical)")
                        text_prompt_2 = gr.Textbox(value="", label="Text Prompt 2", info="Overrides Audio Prompt 2")
            interp_default=0.5
            interp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=interp_default, label="Interpolation", info="Prompt 1  <--->  Prompt 2")
            
            tab1.select(fn=partial(set_active_tab_info,"tab1"))
                        
        with gr.Tab("Audio Algebra!") as tab2: 
            with gr.Row():
                with gr.Box():
                    audio_a = gr.Audio(label="Audio Prompt A", type="filepath", elem_id="aud_comp1", )
                    text_prompt_a = gr.Textbox(value="", label="Text Prompt A", info="Overrides Audio Prompt A")
                    weight_a = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=1.0, label="Weight A")
                with gr.Box():
                    audio_b = gr.Audio(label="Audio Prompt B", type="filepath", elem_id="aud_comp1", )
                    text_prompt_b = gr.Textbox(value="", label="Text Prompt B", info="Overrides Audio Prompt B")
                    weight_b = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=-1.0, label="Weight B")
                with gr.Box():
                    audio_c = gr.Audio(label="Audio Prompt C", type="filepath", elem_id="aud_comp1", )
                    text_prompt_c = gr.Textbox(value="", label="Text Prompt C", info="Overrides Audio Prompt C")
                    weight_c = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=1.0, label="Weight C")

            tab2.select(fn=partial(set_active_tab_info,"tab2"))
      
        with gr.Box():
            with gr.Column():
                init_audio = gr.Audio(label="Init Audio", type="filepath", elem_id="aud_comp3")
                init_strength_default=0.7
                init_strength_slider = gr.Slider(minimum=0.01, maximum=0.99, step=0.01, value=init_strength_default, label="Init Strength", info="'Wet'  <--->  'Dry'")
                    
        with gr.Box():
            with gr.Row():
                batch_slider = gr.Slider(minimum=1, maximum=8, value=1, step=1, label="Variations", info="Number of variations")
            #crossfade = gr.Slider(minimum=0, maximum=4, value=0, step=0.5, label="Cross-fade variations (s)")
                with gr.Column(scale=0, min_width=110):
                    crossfade = gr.Checkbox(value=True, label="Crossfade", elem_id="check_comp", info="Crossfade variations")
                cfg_default=4
                cfg_slider = gr.Slider(minimum=-5, maximum=50, step=0.5, value=cfg_default, label="CFG Scale", info="2 to 6, or...?")
                with gr.Column(scale=0, min_width=110):
                    seed = gr.Number(label="Seed", value=np.random.randint(low=0, high=10**6-1) , precision=0)

        #model_select = gr.Radio(model_choices, value=(model_choices[0] if model_choice is None else model_choice), label="Model choice")
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    clear_btn = gr.Button("Clear")
                    save_btn  = gr.Button("Save")
                with gr.Column():
                    submit_btn = gr.Button("G  E  N  E  R  A  T  E", variant='primary', elem_id="submit")
           
            
        output_audio = gr.Audio(label="Output Audio", elem_id="aud_comp4", type="filepath",) 
                    
        with gr.Row():    #     label="Outputs"):
            embed_plot = gr.Plot(label="Latents 3DPCA.  Color = Time: Dark -> Light", value=harmonai_logo_3d_plotly())

        # button actions
        inputs = [audio_1, text_prompt_1, audio_2, text_prompt_2, 
                  interp_slider, init_audio, init_strength_slider,
                  cfg_slider, seed, batch_slider, crossfade, 
                  audio_a, text_prompt_a, weight_a, 
                  audio_b, text_prompt_b, weight_b,
                  audio_c, text_prompt_c, weight_c,
                 ]
        outputs = [output_audio, embed_plot]
        wrapper = partial(process_audio, device=device, verbose=verbose, show_embeddings=True) # package non-Gradio args into a single function
        submit_btn.click(fn=wrapper, inputs=inputs, outputs=outputs)

        reset_vals = [None, "", None, "", interp_default, None, init_strength_default, cfg_default, np.random.randint(low=0, high=10**6-1) , 1,  None,  None, harmonai_logo_3d_plotly()]
        clear_btn.click(lambda: reset_vals, outputs=inputs+outputs)
        save_btn.click(lambda *objects:print(objects), inputs=inputs) 
            
        with gr.Row():
            example_gui = gr.Examples(fn=wrapper, examples = get_examples(), inputs=inputs,  outputs=outputs, 
                        label="Examples: Click on a row to populate Inputs above, then press GENERATE",
                        #cache_examples=cache_examples, run_on_click=True, # Gradio bug in Blocks means these don't work
                        )
        
    if verbose: print("\nLaunching GUI.")
    
    auth = ( os.getenv('MIRAGE_USERNAME', ''), os.getenv('MIRAGE_PASSWORD', '') )
    
    
    _, _, public_link = demo.queue().launch(
        share=public, prevent_thread_lock=True, show_error=True, auth=auth, favicon_path='harmonai_logo.png') 

    save_html_hosting_info(public_link)

    if verbose: print("Entering sleep loop...")
    while True: sleep(5)

    
    

def save_html_hosting_info(share_url,
                            info_file="/fsx/shawley/info/mirage.html",
                            host_url="https://hedges.belmont.edu/mirage/"):
    share_url += '?__theme=dark' # force dark mode
    HTML_string = (
    """
    <DOCTYPE html>
    <html>
        <head>
        <title>MIRAGE Demo</title>
        <meta charset="UTF-8" />"""
        f'\n    <meta property="og:url" content="{host_url}">'
        f'\n    <meta property="og:image" content={host_url}mirage_screenshot.png">'
        """
        <meta property="og:title" content="Demo of MIRAGE">
        <meta property="og:description" content="Music Information Retrieval-based Audio Generation via Entropy">
        """
        f'\n    <meta http-equiv="Refresh" content="2; url={share_url}" />'
        """
        </head>
        <body>
        <h1>Redirecting</h1>
        Redirecting in 2 seconds.  If you are not automatically redirected, """ 
        f'click <a href="{share_url}">here</a>.'
        """
        </body>
        </html>"""
    )
    text_file = open(info_file, "w")
    text_file.write(HTML_string)
    text_file.close()
    return HTML_string






if __name__ == '__main__':
    app_name = Path(sys.argv[0]).stem # whatever we're going to call this script
    #print(f"Running {app_name}...")

    model_choices = ['CLAPDAE-22s','CLAPDAE-22s-fp16']

    parser = argparse.ArgumentParser(description=f"{app_name}", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cache', action='store_true', help="Cache examples (takes time on startup but they execute fast)")
    parser.add_argument("-f", "--filename", type=str, help="Audio file to process. If none given, then runs GUI.", default="")
    parser.add_argument('-g', '--gui', action='store_true', help="Run GUI even if filename was specified")
    parser.add_argument('-i', '--init', action='store_true', help="Initialize/load model before doing other things (otherwise waits for audio passed in")
    parser.add_argument("-m", "--model", type=str, default=model_choices[0], help=f"Model name to use, one of {model_choices}")
    parser.add_argument("-o", "--output", type=str, help="Output audio file (default is [filename_stem]_processed.wav)", default="")
    parser.add_argument("-s", "--sr", type=int, default=48000, help="Sample rate at which to resample audio file")
    parser.add_argument('-p', '--public', action='store_true', help="Run on public IP address (for Gradio sharing)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print status/debugging messages")
    parser.add_argument('-V', '--version', action='store_true', help="Print version info and exit")
    args = parser.parse_args()
    if args.version: 
        print(app_name, __version__)
        sys.exit(0)

    device = get_device()  # cuda, mps, cpu, etc
    if args.verbose: print("device =", device)
    half_precision = 'fp16' in args.model

    
    if args.init or args.filename: 
        if args.verbose: print("\nInitializing model...")
        get_model_ready(args.model, device, verbose=args.verbose) 

    if args.filename and (app_name != 'ipykernel_launcher'):  # if running in Jupyter cell, launch GUI
        try: 
            print(f"\nLoading audio file {args.filename}...")
            waveform = load_audio(args.filename, sr=args.sr)
            sr, new_waveform = process_audio((args.sr, waveform), args.model, verbose=args.verbose, device=device)
            out_filename = args.output if args.output else Path(args.filename).stem+"_processed.wav"
            print(f"Saving processed audio to {out_filename}")   
            torchaudio.save(out_filename, new_waveform, args.sr)  # btw torchaudio can write mp3s  :)
        except Exception as e:
            print(f"TODO: CLI has not kept pace with GUI development, need to update. Sorry. \n  ERROR processing audio file {args.filename}:")
            print(str(e).replace(r'\n', '\n').replace("'","").strip())
            
    if args.gui or (not args.filename):
        #print("Running GUI...")
        run_gui(device, model_choice=args.model,
            title="MIRAGE", verbose=args.verbose, public=args.public, model_choices=model_choices,
            cache_examples=args.cache,
            description="Music Information Retrieval-based Audio Generation via Entropy. <br><br>If this demo dies, reload <a href='https://hedges.belmont.edu/mirage/'>https://hedges.belmont.edu/mirage/</a>")
