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

# args = {
#         "min-chunk-size" : 1.0,
#         "model_cache_dir" : None,
#         "model_dir" : None,
#         "lan" : 'en',
#         "task" : 'transcribe',
#         "backend" : "faster-whisper",
#         "buffer_trimming" : "segment",
#         "buffer_trimming_sec" : 15,
#         }

# args = dotdict(args)


WORDS = ''
def clear_words():
    global WORDS
    while True:
        time.sleep(30)  # Wait for 30 seconds
        WORDS = ''  # Clear the WORDS variable

# Start the thread
threading.Thread(target=clear_words).start()
class ServerProcessor:

    def __init__(self, online_asr_proc : OnlineASRProcessor, min_chunk, real_time):
        self.online_asr_proc = online_asr_proc
        self.online = online_asr_proc
        
        self.min_chunk = 1 #min_chunk TODO: change this to min_chunk
        # self.t = ''
        self.last_end = None
        self.real_time = real_time  

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

        return y


    def process(self, audio):
        global WORDS
        self.online_asr_proc.init()
        a = self.t_receive_audio_chunk(audio) if self.real_time else self.p_receive_audio_chunk(audio)

        self.online_asr_proc.insert_audio_chunk(a)
        inc = self.online.process_iter()
        WORDS += inc
        return WORDS


class ASRTranscriber:
    def __init__(self):
        self.asr = None
        self.online = None
        self.current_model = None
        
        self.curr_vad = False
        self.curr_RT = True

    def transcribe(self, audio, model, vad, real_time):
        if model != self.current_model:
            # Only reinitialize the ASR and processor if the model has changed
            t = time.time()
            print(f"Loading Whisper {size} model for {language}...",file=sys.stderr,end=" ",flush=True)
            self.asr = FasterWhisperASR(modelsize=model, lan='en', cache_dir=None, model_dir=None)
            if self.curr_vad != vad:
                print(f"setting VAD filter to {args.vad}",file=sys.stderr)
                self.curr_vad = vad
                if vad:
                    self.asr.use_vad()
            e = time.time()
            print(f"done. It took {round(e-t,2)} seconds.",file=sys.stderr)
            tokenizer = None
            self.online = OnlineASRProcessor(self.asr, tokenizer, buffer_trimming=('segment', 15))
            self.current_model = model
        if self.curr_RT != real_time:
            self.curr_RT = real_time

        proc = ServerProcessor(self.online, min_chunk=1.0 , real_time=real_time)
        result = proc.process(audio)
        return result

transcriber = ASRTranscriber()

demo = gr.Interface(
    fn=transcriber.transcribe, 
    inputs=[
        gr.Audio(sources=["microphone"], streaming=True),
        gr.Textbox(value="tiny", placeholder = "tiny", label="Model"),
        gr.Checkbox(value=False, label="VAD"),
        gr.Checkbox(value=True, label="Real Time"),
    ],
    outputs="text",
    live=True
)

demo.launch(debug=True)


demo.launch(debug=True)
