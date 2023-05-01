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
import gradio as gr
import traceback
import math
from functools import partial
from aeiou.core import load_audio, get_device
print("   Starting CLAPDAE import")
from audio_algebra.given_models import CLAPDAE 
#import plotly.express as px 
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
print("Finished with imports.")
from time import sleep
import random
from einops import rearrange

model, current_model_choice = None, '' # global vars to keep (same) model initialized thru multiple GUI ops 
half_precision = False 

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
                      fade_type='linear'): # 'linear'|'sine'
    "flattens but cross-fades between batch elements.  Source: @drscotthawley ;-) "
    if cl==0 or ab.shape[0]==1: return ab
    assert len(ab.shape)==3, f"Expecting batched audio but got ab.shape = {ab.shape}"
    fade_in = torch.linspace(0, 1, steps=cl, dtype=ab.dtype, device=ab.device) 
    if fade_type=='sine': fade_in = torch.sin(fade_in * math.pi/2 )    # typically unused but why not
    ab[1:,:,:cl] = ab[1:,:,:cl]*fade_in + ab[:-1,:,-cl:]*(1-fade_in)   # apply cross-fades
    chopped_flattened = rearrange( ab[:,:, :-cl], 'b c n -> c (b n)' ) # chop off fade-out regions (and last piece)
    return torch.cat( ( chopped_flattened, ab[-1,:,-cl:]), dim=-1 )    # stick last piece back on 

    
@torch.no_grad()    
def process_audio(
    device,           # pytorch device to use
    audio_tup=None,        # a tuple of (sr, waveform) where sr is sample rate
    text_prompt="",
    audio_tup2=None,  # optional 2nd audio for interpolation
    text_prompt2="",
    interp_scale=0.5, # relative strength of audio_tup & audio_tup2
    batch_size=1,
    cfg_scale=4.0,    # cfg scale, a number
    #model_choice='CLAPDAE-22s', # a string, the name of the model to use
    crossfade_secs=0.0,
    init_audio_tup=None, 
    init_strength=0.4,
    verbose=True, # how much info we print while executing
    show_embeddings=False, # whether to show the embeddings plot in the output - only in GUI mode
    model_choice='CLAPDAE-22s', # a string, the name of the model to use
    ):
    """That code which processes the audio"""
    global half_precision
    
    if verbose:
        print(f"\n --- process_audio: passed in:  device = {device}, audio_tup = {audio_tup}, cfg_scale = {cfg_scale}, model_choice = {model_choice}, audio_tup2 = {audio_tup2}, init_audio_tup = {init_audio_tup}")
    
    if audio_tup is None and not text_prompt: 
        raise gr.Error("No audio or text prompt passed in.") 
        return audio_tup 
    if cfg_scale is None: 
        raise gr.Error("No cfg_scale given.") 
        return None, None
    
    
    half_precision = 'fp16' in model_choice
    
    try:
        batch_size = int(batch_size)
        if verbose: print("process_audio: calling unpack_audio_tup")
        waveform, audio_info = unpack_audio_tup(audio_tup, verbose=verbose)
        waveform2, audio_info2 = unpack_audio_tup(audio_tup2, verbose=verbose)  # returns None, None if not given
        init_waveform, init_info = unpack_audio_tup(init_audio_tup, verbose=verbose)  # returns None, None if not given
        
        get_model_ready(model_choice, device, verbose=verbose) # if needed
        
        ##---------------- Do the actual audio processing ------------------
        if waveform is not None: waveform = waveform.to(model.device)
        bypass_flip = False  # for testing gradio i/o faster: skips the model and just flips the audio 
        if bypass_flip: 
            new_waveform = waveform.flip(dims=[1]) # just a flip op
        else:    
            if verbose: print("\n-------  Now processing audio via model ---------")
            with torch.no_grad():    # turn off gradients
                
                
                if verbose: print(f"  ------ Getting CLAP Embeddings:")
                embeddings   = model.embed(waveform) if waveform is not None else None
                if text_prompt:
                    embeddings = model.embed(text_prompt) # text prompt supersedes audio 1 (bc currently lazy)
                if (waveform2 is not None) or text_prompt2:  # interpolation
                    embeddings2  = model.embed(text_prompt2) if text_prompt2 else model.embed(waveform2)
                    print("      --- Interpolating CLAP Embeddings")
                    interp_type='Spherical', # interpolation method 
                    embeddings = interp_embeddings(embeddings, embeddings2, interp_scale, interp_type=interp_type)     
                embeddings  = half_it(embeddings) # half only works on the decode side b/c CLAP doesn't like Half
                
                
                init_audio_latents = None
                print("        init_waveform = ",init_waveform)
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
                cfg_scales = [cfg_scale] # TODO: only allow 1 cfg for no, plan to add more
                crossfade_len=int(crossfade_secs*48000) # TODO: hardcoding sample rate here
                if verbose: print(f"\n  ------ Generating Audio, cfg_scales = {cfg_scales}, batch_size = {batch_size}, init_strength = {init_strength}")
                fakes, fake_latents = model.generate(embeddings, cfg_scales=cfg_scales, 
                                                     init_audio_latents=init_audio_latents, 
                                                     init_strength=init_strength, 
                                                     batch_size=batch_size,
                                                     crossfade_len=crossfade_len)
                fakes = crossfade_flatten(fakes, crossfade_len)
                    
                if verbose: print("\n  ------ Finished generating.  fakes ",sdd_str(fakes))
        ##----------------------Finished with true audio processing--------------------------
        # now get it ready to be returned to gradio or wherever inputs came from 
        
        if audio_info is None:  # supply gradio defaults if generated from text prompt
            audio_info = {'sr':48000, 'mono_in':False, 'tensor_in':False, 'int_in':True, 'channels_first':False}
            
        new_audio_tup = repack_audio_tup(fakes, audio_info, verbose=verbose)

        if show_embeddings:
            if verbose: print("Plotting embeddings graph")
            fig = pca_point_cloud(fake_latents.float(), output_type='plotly', mode='lines+markers', color_scheme='')
        else:
            fig = None
    except Exception as e:   
        msg = f"Error processing audio: {e}\n"+traceback.format_exc()
        raise gr.Error(msg) # either shows in GUI or gets printed later in CLI
    if show_embeddings:
        return (new_audio_tup, fig)
    else:
        return new_audio_tup


def do_clear(args): #???
    for x in args:
        print("x = ",x)
        try:
            x.clear()
        except:
            pass
        

def run_gui(device, verbose=True, public=False, model_choices=["model1", "model2", "model3"], model_choice=None, **kwargs):
    """Launches the GUI"""

    css_code='body{background-image:url("harmonai_logo_big.png");}'

    with gr.Blocks(title=kwargs['title'], css=css_code) as demo: 
        toptext = gr.HTML(f"<center><h1>{kwargs['title']}</h1>{kwargs['description']}</center>")
        with gr.Row():   #     label="Input 1"):
            with gr.Box():
                with gr.Column():
                    audio1 = gr.Audio(label="1st Audio Prompt")
                    textprompt1 = gr.Textbox(value="", label="Text Prompt 1 (Takes precedence over 1st Audio Prompt)")
            with gr.Box():
                with gr.Column():   # label="Model Controls"):
                    audio2 = gr.Audio(label="Optional: 2nd Audio Prompt for interpolation")
                    #interp_type = gr.Radio(['Spherical', 'Linear'], value='Spherical', label="Interpolation type (Leave it on Spherical)")
                    textprompt2 = gr.Textbox(value="", label="Text Prompt 2 (Takes precedence over 2nd Audio Prompt)")

        interp_slider = gr.Slider(minimum=-0.05, maximum=1.05, value=0.5, label="Interpolation scale (0 = all 1st prompt, 1 = all 2nd prompt")
        with gr.Row():
            with gr.Box():
                with gr.Column():     #     label="Optional Init Audio"):
                    init_audio = gr.Audio(label="Optional: Init audio")
                    init_str_slider = gr.Slider(minimum=0.01, maximum=0.99, value=0.7, label="Strength of init audio (You probably want it high)")

            with gr.Box():
                with gr.Column():
                    batch_slider = gr.Slider(minimum=1, maximum=8, value=1, step=1, label="Number of variations (strung along as one output)")
                    cfg_slider = gr.Slider(minimum=-5, maximum=50, value=4, label="CFG scale (Typically 2 to 6, but go crazy)")
                    #model_select = gr.Radio(model_choices, value=(model_choices[0] if model_choice is None else model_choice), label="Model choice")
                    crossfade_secs = gr.Slider(minimum=0, maximum=4, value=0, step=0.5, label="Cross-fade between variations (in seconds)")
                    
                    submit_btn = gr.Button("Submit", variant='primary')
                    clear_btn = gr.Button("Clear (Doesn't work right now)")

        with gr.Row():    #     label="Outputs"):
            output_audio = gr.Audio(label="Output audio")
            embed_plot = gr.Plot(label="Embeddings 3DPCA")

        inputs = [audio1, textprompt1, audio2, textprompt2, interp_slider, batch_slider,
                        cfg_slider, crossfade_secs, init_audio, init_str_slider]
        outputs = [output_audio, embed_plot]
        wrapper = partial(process_audio, device, verbose=verbose, show_embeddings=True) # package non-Gradio args into a single function
        submit_btn.click(fn=wrapper, inputs=inputs, outputs=outputs)
        clear_btn.click(lambda: None, None, audio1 ) # not sure how to get this to work

    simple_interface = False
    if simple_interface: # overwrite complicated demo above
        demo = gr.Interface(fn=wrapper, 
                inputs=[
                        gr.Audio(label="1st Audio Prompt"),
                        gr.Textbox(value="", label="Text Prompt 1 (Takes precedence over 1st Audio Prompt)"),
                        gr.Audio(label="Optional: 2nd Audio Prompt for interpolation"),
                        gr.Textbox(value="", label="Text Prompt 2 (Takes precedence over 2nd Audio Prompt)"),
                        #gr.Radio(['Spherical', 'Linear'], value='Spherical', label="Interpolation type (Leave it on Spherical)"),
                        gr.Slider(minimum=-0.05, maximum=1.05, value=0.5, label="Interpolation scale (0 = all 1st audio, 1 = all 2nd audio"),
                        gr.Slider(minimum=1, maximum=8, value=2, step=1, label="Number of variations (strung along as one output)"),
                        gr.Slider(minimum=-5, maximum=50, value=4, label="CFG scale (Typically 2, 4, or 6, but we'll let the range be a bit crazy for testing)"),
                        gr.Radio(model_choices, 
                                 value=(model_choices[0] if model_choice is None else model_choice), 
                                 label="Model choice"),
                        gr.Audio(label="Optional: Init audio"),
                        gr.Slider(minimum=0.01, maximum=0.99, value=0.7, label="Strength of init audio (You probably want it high)"),
                        
                        ],
                outputs=[gr.Audio(label="Output audio"),
                         gr.Plot(label="Embeddings 3DPCA")], **kwargs)
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
    HTML_string = (
    """
    <DOCTYPE html>
    <html>
        <head>
        <title>(MI)RAGE Demo</title>
        <meta charset="UTF-8" />"""
        f'    <meta http-equiv="Refresh" content="2; url={share_url}" />'
        f'    <meta property="og:url" content="{host_url}">'
        f'    <meta property="og:image" content={host_url}/mirage_screenshot.png">'
        """
        <meta property="og:title" content="Demo of (MI)RAGE">
        <meta property="og:description" content="(Music Information) Retrieval-based Audio Generation via Entropy">
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
            sr, new_waveform = process_audio(device, (args.sr, waveform), args.model, verbose=args.verbose)
            out_filename = args.output if args.output else Path(args.filename).stem+"_processed.wav"
            print(f"Saving processed audio to {out_filename}")   
            torchaudio.save(out_filename, new_waveform, args.sr)  # btw torchaudio can write mp3s  :)
        except Exception as e:
            print(f"ERROR processing audio file {args.filename}:")
            print(str(e).replace(r'\n', '\n').replace("'","").strip())
            
    if args.gui or (not args.filename):
        #print("Running GUI...")
        run_gui(device, model_choice=args.model,
            title="(MI)RAGE", verbose=args.verbose, public=args.public, model_choices=model_choices,
            #description="Music Information Retrieval-based Autoencoder for Generation via Entropy")
            description="(Music Information) Retrieval-based Audio Generation via Entropy. <br><br>If this demo dies, reload <a href='https://hedges.belmont.edu/mirage/'>https://hedges.belmont.edu/mirage/</a>")
