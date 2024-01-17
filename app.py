import gradio as gr
import numpy as np
import sys
import os
import time
from whisper_online import *
import io
import soundfile as sf
import audioop
import threading

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
        "model" : os.environ.get("model", "tiny"),
        "model_cache_dir" : None,
        "model_dir" : None,
        "lan" : 'en',
        "task" : 'transcribe',
        "backend" : "faster-whisper",
        "vad" : bool(os.environ.get("vad", False)),
        "buffer_trimming" : "segment",
        "buffer_trimming_sec" : 15,
        "real_time" : bool(os.environ.get("real_time", True)),
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


print(f"setting VAD filter to {args.vad}",file=sys.stderr)
if args.vad:
    ASR.use_vad()
TOKENIZER = None
ONLINE = OnlineASRProcessor(ASR,TOKENIZER,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
REAL_TIME = args.real_time
WORDS = ''
def clear_words():
    global WORDS
    while True:
        time.sleep(30)  # Wait for 30 seconds
        WORDS = ''  # Clear the WORDS variable

# Start the thread
threading.Thread(target=clear_words).start()
class ServerProcessor:

    def __init__(self, online_asr_proc, min_chunk):
        self.online_asr_proc = online_asr_proc
        self.min_chunk = 1 #min_chunk TODO: change this to min_chunk
        # self.t = ''
        self.last_end = None

    def p_receive_audio_chunk(self, audio):
        # Convert the audio file path to audio data
        audio_data = load_audio_chunk(audio, 0, 1)
        return audio_data
    
    def t_receive_audio_chunk(self, new_chunk):
        
        sr, y = new_chunk
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
            
        y = audioop.ratecv(y, 2, 1, sr, 16000, None)[0]
        y = np.frombuffer(y, dtype=np.float32)

        # print(f"changed y to become \n {y} \n", file=sys.stderr, flush=True)
        return y


    def process(self, audio):
        global WORDS
        self.online_asr_proc.init()
        a = self.t_receive_audio_chunk(audio) if REAL_TIME else self.p_receive_audio_chunk(audio)
        # if a is None:
        #     print("break here", file=sys.stderr, flush=True)
        #     return
        self.online_asr_proc.insert_audio_chunk(a)
        inc = ONLINE.process_iter()
        WORDS += inc
        return WORDS


def transcribe(audio):

    proc = ServerProcessor(ONLINE, min_chunk = args.min_chunk_size)
    result = proc.process(audio)
    return result

demo = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources=["microphone"], streaming=True),
    outputs="text",
    live=True
)

demo.launch(debug=True)
