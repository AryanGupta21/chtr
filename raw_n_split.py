import pandas as pd
import torch, torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
import os

TARGET_SR = 16000
CLIP_SEC = 4.0
NUM_SAMPLES = int(TARGET_SR * CLIP_SEC)

AUDIO_DIR = "ICBHI_final_database"  # folder with audio files

def load_resample(path, sr=TARGET_SR):
    wav, s = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = T.Resample(s, sr)(wav)
    wav = wav.float()  # ensure float32 dtype
    return wav

def pad_trim(wav, n):
    N = wav.shape[-1]
    if N == n:
        return wav
    if N > n:
        return wav[:, :n]
    return torch.nn.functional.pad(wav, (0, n - N))

def wav_to_fixed_raw(filename):
    path = os.path.join(AUDIO_DIR, filename)
    assert os.path.isfile(path), f"File not found: {path}"
    wav = load_resample(path, TARGET_SR)  # (1, N)
    wav = pad_trim(wav, NUM_SAMPLES)
    rms = torch.sqrt(torch.mean(wav ** 2) + 1e-8)
    return (wav / (rms + 1e-8)).float()  # normalize and return

# Read CSV
df = pd.read_csv("icbhi.csv")

# Create 'raw' folder if it doesn't exist
if not os.path.exists("raw"):
    os.makedirs("raw")

# Split dataset (no stratify assumed for multi-label)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=43)

# Process and save raw waveform tensors for train samples
for idx, row in train_df.iterrows():
    filename = row["filename"]
    raw_tensor = wav_to_fixed_raw(filename)
    save_path = os.path.join("raw", f"{os.path.splitext(filename)[0]}_raw.pt")
    torch.save(raw_tensor, save_path)
    if idx == 0:
        labels = row[["crackle", "wheeze"]].values.astype(int)
        print(f"Saved {save_path} with labels {labels}")
        # Optional break after first sample for debug/demo
        break
