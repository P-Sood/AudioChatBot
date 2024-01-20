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
from transformers import pipeline
import torch
from huggingface_hub import login

LLAMA_TOK = os.environ.get('read')
login(LLAMA_TOK)

WORDS = ''
def clear_words():
    global WORDS
    while True:
        time.sleep(60)  # Wait for 60 seconds
        WORDS = ''  # Clear the WORDS variable

# Start the thread
# threading.Thread(target=clear_words).start()
class ServerProcessor:

    def __init__(self, online_asr_proc : OnlineASRProcessor, min_chunk):
        self.online_asr_proc = online_asr_proc
        self.online = online_asr_proc
        
        self.min_chunk = 1 #min_chunk TODO: change this to min_chunk
        self.t = ''
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

        return y


    def process(self, audio , text_gen):
        global WORDS
        self.online_asr_proc.init()
        a = self.t_receive_audio_chunk(audio) if REAL_TIME else self.p_receive_audio_chunk(audio)

        self.online_asr_proc.insert_audio_chunk(a)
        inc = self.online.process_iter()
        WORDS += inc
        print(f"WORDS: {WORDS}",file=sys.stderr, flush=True)
        if '?' in inc:
            print("QUESTION DETECTED",file=sys.stderr, flush=True)
            sequences = text_gen(
                    WORDS, 
                    temperature=0.9, 
                    top_k=50, 
                    top_p=0.9,
                    do_sample=True,
                    max_length=500
                    )

            print(f"sequences generated \n {sequences}",file=sys.stderr, flush=True)
            for seq in sequences:
                self.t += seq['generated_text']
            WORDS = ''
            return self.t
            
        return self.t

class ASRTranscriber:
    def __init__(self):
        self.asr = None
        self.online = None
        self.current_whisper_model = None
        self.current_text_model = None
        
        self.p = None
        
        self.curr_vad = False
        

    def transcribe(self, audio, whisper_model, vad , text_model):
        if whisper_model != self.current_whisper_model:
            # Only reinitialize the ASR and processor if the whisper_model has changed
            t = time.time()
            self.asr = FasterWhisperASR(modelsize=whisper_model, lan='en', cache_dir=None, model_dir=None)
            self.current_whisper_model = whisper_model
            if self.curr_vad != vad:
                print(f"setting VAD filter to {args.vad}",file=sys.stderr)
                self.curr_vad = vad
                if vad:
                    self.asr.use_vad()
            e = time.time()
            print(f"load whisper. It took {round(e-t,2)} seconds.",file=sys.stderr)
            self.online = OnlineASRProcessor(self.asr, tokenizer = None, buffer_trimming=('segment', 15))
        if text_model != self.current_text_model:
            self.current_text_model = text_model
            t = time.time()
            
            print("before loading llama.",file=sys.stderr, flush=True)
            self.p = pipeline("text-generation", 
                                 model=text_model,
                                #  torch_dtype=torch.float32, 
                                 )
            e = time.time()
            print(f"loaded llama. It took {round(e-t,2)} seconds.",file=sys.stderr)
            
        proc = ServerProcessor(self.online, min_chunk=1.0 )
        result = proc.process(audio , self.p)
        return result

transcriber = ASRTranscriber()
REAL_TIME = True
demo = gr.Interface(
    fn=transcriber.transcribe, 
    inputs=[
        gr.Audio(sources=["microphone"], streaming=True),
        gr.Radio(['tiny.en','tiny','base.en','base','small.en','small','medium.en','medium','large-v1','large-v2','large-v3','large'], info="Turn on the audio recording before changing me. Allow from 2 to 29 seconds for me to load models" , value = "medium.en" , label="WhisperModel" , interactive=True),
        gr.Checkbox(value=False, label="VAD" , info="Turn on the audio recording before changing me. Make sure to stop the recording to check out the transcription as it can get buggy.\n I also remove the transcription after 30 seconds so you can get a fresh output to try new things on"),
        # gr.Radio(['meta-llama/Llama-2-7b-chat-hf','meta-llama/Llama-2-13b-chat-hf','meta-llama/Llama-2-70b-chat-hf'], info="Turn on the audio recording to load the models in. Allow 2-3 minutes to load the model. I dont recommend changing it, it takes so long to switch models" , value = "meta-llama/Llama-2-7b-chat-hf" , label="TextModel" , interactive=True),
        gr.Radio(['microsoft/phi-2' , 'ahxt/LiteLlama-460M-1T'], info="Turn on the audio recording to load the models in. Allow 2-3 minutes to load the model. I dont recommend changing it, it takes so long to switch models" , value = "ahxt/LiteLlama-460M-1T" , label="TextModel" , interactive=True),
    ],
    outputs="text",
    live=True
)

demo.launch(debug=True)
