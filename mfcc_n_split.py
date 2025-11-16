import pandas as pd
import torch, torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
import os

TARGET_SR = 16000
N_MFCC = 30
FRAME_MS, HOP_MS = 25, 10
WIN = int(TARGET_SR * FRAME_MS / 1000)
HOP = int(TARGET_SR * HOP_MS / 1000)
mfcc_tf = T.MFCC(
    sample_rate=TARGET_SR,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 512, "win_length": WIN, "hop_length": HOP, "n_mels": 64, "f_min": 0.0, "f_max": TARGET_SR / 2}
)

AUDIO_DIR = "ICBHI_final_database"  # folder with audio files

def load_resample(path, sr=TARGET_SR):
    wav, s = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = T.Resample(s, sr)(wav)
    wav = wav.float()
    return wav

def wav_to_mfcc(filename):
    path = os.path.join(AUDIO_DIR, filename)
    assert os.path.isfile(path), f"File not found: {path}"
    wav = load_resample(path, TARGET_SR)
    x = mfcc_tf(wav)  # (n_mfcc, T)
    m, s = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((x - m) / s).float()

# Read CSV
df = pd.read_csv("icbhi.csv")

# Create mfcc folder if not exists
if not os.path.exists("mfcc"):
    os.makedirs("mfcc")

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=43)

# Process and save MFCC tensors for training samples
for idx, row in train_df.iterrows():
    filename = row["filename"]
    mfcc_tensor = wav_to_mfcc(filename)
    save_path = os.path.join("mfcc", f"{os.path.splitext(filename)[0]}_mfcc.pt")
    torch.save(mfcc_tensor, save_path)
    if idx == 0:
        labels = row[["crackle", "wheeze"]].values.astype(int)
        print(f"Saved {save_path} with labels {labels}")
        # Optional break after first file for quick check
        break
