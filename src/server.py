import os
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["neutral", "happy", "sad", "anger", "fear", "disgust"]
LOCAL_MODEL_PATH = "exports/whisper_lora_merged" # The folder with your trained weights
EMO_HEAD_PATH = os.path.join(LOCAL_MODEL_PATH, "emo_head.pt")

print(f"Using device: {DEVICE}")

# 1. Load the STANDARD Processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 2. Load the STANDARD Brain (For Speech-to-Text / Any word detection)
print("Loading standard OpenAI Whisper for ASR...")
model_asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(DEVICE)

# 3. Load the EMOTIONAL Brain (Your merged LoRA model for Emotion detection)
print("Loading your fine-tuned Emotional Brain for SER...")
model_emo = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)

# 4. Setup your CUSTOM Emotion Head
print("Setting up Custom Emotion Head...")
d = model_emo.config.d_model # 768
emo_head = nn.Sequential(
    nn.LayerNorm(d), 
    nn.Linear(d, d // 2), 
    nn.GELU(),
    nn.Dropout(0.1), 
    nn.Linear(d // 2, len(EMOTIONS))
)

if os.path.exists(EMO_HEAD_PATH):
    emo_head.load_state_dict(torch.load(EMO_HEAD_PATH, map_location=DEVICE))
else:
    print(f"ERROR: {EMO_HEAD_PATH} not found!")

emo_head.to(DEVICE)

# Set everything to evaluation mode
model_asr.eval()
model_emo.eval()
emo_head.eval()

app = FastAPI()

@app.post("/cast_spell")
async def cast_spell(audio_file: UploadFile):
    with open("temp.wav", "wb") as f:
        f.write(await audio_file.read())
    
    wav, sr = torchaudio.load("temp.wav")
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0, keepdim=True)
    
    # Volume Normalization (Helps the SER model stay consistent)
    if wav.abs().max() > 0:
        wav = wav / wav.abs().max()

    input_features = processor(
        wav.squeeze().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(DEVICE)
    
    with torch.no_grad():
        # --- TASK 1: TRANSCRIPTION (Standard Brain) ---
        # This model knows all English words, including "Fireball"
        predicted_ids = model_asr.generate(input_features)
        asr_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()
        asr_text = "".join(c for c in asr_text if c.isalnum() or c.isspace())

        # --- TASK 2: EMOTION (Emotional Brain + Emo Head) ---
        # We get the 'hidden states' from the model you actually trained
        encoder_outputs = model_emo.get_encoder()(input_features)
        last_hidden_state = encoder_outputs.last_hidden_state
        
        # Feed the hidden states into the head
        emo_logits = emo_head(last_hidden_state.mean(dim=1))
        
        probs = F.softmax(emo_logits, dim=-1)[0]
        pred_idx = probs.argmax(-1).item()
        
        # Confidence check: If AI is very unsure, default to neutral
        if probs[pred_idx] < 0.35:
            pred_emo = "neutral"
            emotion_conf = probs[pred_idx].item()
        else:
            pred_emo = EMOTIONS[pred_idx]
            emotion_conf = probs[pred_idx].item()
    
    print(f"🔮 Word: '{asr_text}' | Emotion: {pred_emo} ({emotion_conf*100:.1f}%)")
    
    return {
        "spoken_text": asr_text,
        "emotion": pred_emo,
        "confidence": float(emotion_conf)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)