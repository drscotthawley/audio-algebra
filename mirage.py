#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A CLI or GUI-based demo for processing audio using embeddings. Like Destructo
GUI uses Gradio
CLI based (so Discord-bot-compatible) but will call up GUI if no filename given

Candidate names:
    Simthesizer / Simthesize / Symth 
    MIRAGE: Music Information Retrieval-based Autoencoder for Generation via Entropy
    MIRACLE: Music Information Retrieval-based Audio Creation via Lossy Encoding 
    SPECTER: Sound Production via Efficient Compression Techniques and Entropic Reconstruction
    ReGALE: Reinterpreting Generative Audio via Lossy Encoding 

AUTHOR: Scott H. Hawley @drscotthawley 
LICENSE: Creative Commons 4.0, commercial usage allowed. Just cite this gist URL in a comment to be nice. 

NB: This entire source code can be pasted into a single Juptyer notebook cell and it will run as-is, calling the GUI
"""
print("Starting imports...")
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
import plotly.express as px 
print("Finished with imports.")

model, current_model_choice = None, None # global vars to keep (same) model initialized thru multiple GUI ops 

def unpack_audio_tup(audio_tup, verbose=True):
    """utility routine for process_audio: 
    turn gradio's audio object into something like my models expect:
    My routines want stereo, channels-first, 32-bit float PyTorch tensors, but Gradio gives channels-last int16 numpy arrays
    """
    (sr, waveform) = audio_tup
    mono_in, tensor_in = len(waveform.shape)==1, torch.is_tensor(waveform)
    if not tensor_in: waveform = torch.from_numpy(waveform) 
    int_in = waveform.dtype == torch.int16 # dtype used by gradio
    audio_info = {'sr':sr, 'mono_in':mono_in, 'tensor_in':tensor_in, 'int_in':int_in}
    if verbose: print("audio_info = ",audio_info)
    if int_in:  waveform = waveform.float()/32768.0
    if verbose: print("unpack_audio: waveform.shape, waveform.dtype =",waveform.shape, waveform.dtype)
    if mono_in: 
        waveform = waveform.unsqueeze(0) # add channel dimension if needed
        channels_first = None # undefined
    else:
        channels_first = (waveform.shape[0] < waveform.shape[1])
        audio_info['channels_first'] = channels_first
        if not channels_first: waveform = waveform.transpose(0,1) 
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
    if audio_info['int_in']: new_waveform = torch.tensor(new_waveform * 32768, dtype=torch.int16)
    if not audio_info['tensor_in']: new_waveform = new_waveform.numpy()
    if verbose: print("new_waveform.dtype =", new_waveform.dtype)
    return (audio_info['sr'], new_waveform) 


def get_model_ready(model_choice, device, verbose=True):
    """utility routine for process_audio: gets model ready if it's not already"""
    global model, current_model_choice  # Hate using globals but how else to keep model initialized?

    if verbose: print("current_model_choice, model_choice =", current_model_choice, model_choice)
    if model_choice != current_model_choice:
        if verbose: print("Hang on, we need to load a new model...")
        # TODO: ignoring all other model choices for now, lol. 
        model = CLAPDAE(device=device)
        model.setup()
        current_model_choice = model_choice
    elif model is not None:
        if verbose: print(f"Model's already loaded! {model.__class__.__name__}")
    else:
        raise gr.Error("Can't find the right model, sorry.")


def process_audio(
    device,       # pytorch device to use
    audio_tup,    # a tuple of (sr, waveform) where sr is sample rate
    model_choice, # a string, the name of the model to use
    verbose=True, # how much info we print while executing
    show_embeddings=False, # whether to show the embeddings plot in the output - only in GUI mode
    ):
    """That code which processes the audio"""
    if audio_tup is None: 
        raise gr.Error("No audio passed in.") 
        return audio_tup 
    try:
        waveform, audio_info = unpack_audio_tup(audio_tup, verbose=verbose)        

        get_model_ready(model_choice, device, verbose=verbose) # if needed

        ##---------------- Do the actual audio processing ------------------
        waveform = waveform.to(model.clap_device)
        bypass_flip = False  # for testing gradio i/o faster: skips the model and just flips the audio 
        if bypass_flip: 
            new_waveform = waveform.flip(dims=[1]) # just a flip op
        else:    
            with torch.no_grad():
                embeddings   = model.encode(waveform) 
                new_waveform = model.decode(embeddings)
        #new_waveform *= math.sqrt(-1) # only used to test error handling
        new_waveform = new_waveform.cpu()
        ##------------------------------------------------------------------

        new_audio_tup = repack_audio_tup(new_waveform, audio_info, verbose=verbose)

        if show_embeddings:
            if verbose: print("trying embeddings")
            df = px.data.iris()
            fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
        else:
            fig = None
    except Exception as e:   
        msg = f"Error processing audio: {e}\n"+traceback.format_exc()
        raise gr.Error(msg) # either shows in GUI or gets printed later in CLI
    if show_embeddings:
        return (new_audio_tup, fig) 
    else:
        return new_audio_tup



def run_gui(device, verbose=True, public=False, model_choices=["model1", "model2", "model3"], **kwargs):
    """Launches the GUI"""
    wrapper = partial(process_audio, device, verbose=verbose, show_embeddings=True) # package non-Gradio args into a single function
    demo = gr.Interface(fn=wrapper, 
        inputs=[gr.Audio(label="Input audio"), 
                gr.Radio(model_choices, value=model_choices[0], label="Model choice")], 
        outputs=[gr.Audio(label="Output audio"),gr.Plot(label="Embeddings 3DPCA")], **kwargs)
    demo.launch(share=public)
 


if __name__ == '__main__':
    app_name = Path(sys.argv[0]).stem # whatever we're going to call this script
    #print(f"Running {app_name}...")

    model_choices = ['CLAPDAE']

    parser = argparse.ArgumentParser(description=f"{app_name}", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--filename", type=str, help="Audio file to process. If none given, then runs GUI.", default="")
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

    if args.init: get_model_ready(args.model, device, verbose=args.verbose) 

    if args.filename and (app_name != 'ipykernel_launcher'):  # if running in Jupyter cell, launch GUI
        try: 
            print(f"Loading audio file {args.filename}...")
            waveform = load_audio(args.filename, sr=args.sr)
            sr, new_waveform = process_audio(device, (args.sr, waveform), args.model, verbose=args.verbose)
            out_filename = args.output if args.output else Path(args.filename).stem+"_processed.wav"
            print(f"Saving processed audio to {out_filename}")   
            torchaudio.save(out_filename, new_waveform, args.sr)  # btw torchaudio can write mp3s  :)
        except Exception as e:
            print(f"ERROR processing audio file {args.filename}:")
            print(str(e).replace(r'\n', '\n').replace("'","").strip())
    else:
        print("No filename given; running GUI...")
        run_gui(device, 
            title=app_name.upper(), verbose=args.verbose, public=args.public, model_choices=model_choices,
            description="Music Information Retrieval-based Autoencoder for Generation via Entropy")
