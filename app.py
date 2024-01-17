import gradio as gr
import numpy as np
import sys
import os
import time
from whisper_online import *
import io
import soundfile as sf
# Load your model

#unnecesary comment
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
SAMPLING_RATE = 16000

args = {
        "min-chunk-size" : 1.0,
        "model" : 'tiny',
        "model_cache_dir" : None,
        "model_dir" : None,
        "lan" : 'en',
        "task" : 'transcribe',
        "backend" : "faster-whisper",
        "vad" : False,
        "buffer_trimming" : "segment",
        "buffer_trimming_sec" : 15
        }

args = dotdict(args)

# comment

size = args.model
language = args.lan

t = time.time()
print(f"Loading Whisper {size} model for {language}...",file=sys.stderr,end=" ",flush=True)

ASR = FasterWhisperASR(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
TGT_LANGUAGE = language

e = time.time()
print(f"done. It took {round(e-t,2)} seconds.",file=sys.stderr)


print("setting VAD filter",file=sys.stderr)
ASR.use_vad()
TOKENIZER = None
ONLINE = OnlineASRProcessor(ASR,TOKENIZER,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))



class ServerProcessor:

    def __init__(self, online_asr_proc, min_chunk):
        self.online_asr_proc = online_asr_proc
        self.min_chunk = 1 #min_chunk TODO: change this to min_chunk
        
        self.t = ""

        self.last_end = None

    # def receive_audio_chunk(self, stream, audio):
    #     # Convert the audio file path to audio data
    #     audio_data = load_audio_chunk(audio, 0, 1)
    #     return audio_data
    
    def receive_audio_chunk(self, stream, new_chunk):
        
        sr, y = new_chunk
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
            
        if stream is not None:
            stream = np.concatenate([stream, y])
        else:
            stream = y
        return stream



    def format_output_transcript(self,o):

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            self.t +=  f"{beg} {end}\n"
            print("\n test \n",self.t, "\n" ,file=sys.stderr, flush=True)
            print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg,end,o[2])
        else:
            print(o,file=sys.stderr,flush=True)
            return None

    def process(self, stream, audio):
        self.online_asr_proc.init()
        stream = self.receive_audio_chunk(stream, audio)
        # if stream is None:
        #     print("break here", file=sys.stderr, flush=True)
        #     return
        self.online_asr_proc.insert_audio_chunk(stream)
        o = ONLINE.process_iter()
        return stream , self.format_output_transcript(o)


def transcribe(stream , audio):

    proc = ServerProcessor(ONLINE, min_chunk = args.min_chunk_size)
    stream, result = proc.process(stream , audio)
    return stream, result

demo = gr.Interface(
    fn=transcribe, 
    inputs=["state", gr.Audio(sources=["microphone"], streaming=True)],
    outputs=["state", "text"],
    live=True
)

demo.launch(debug=True)
