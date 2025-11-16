import pandas as pd
import torch, torchaudio
import torchaudio.transforms as T
import os
from tqdm import tqdm

TARGET_SR = 16000
N_MELS = 128
WIN = int(TARGET_SR * 0.025)  # 25ms
HOP = int(TARGET_SR * 0.010)  # 10ms
IMG_H, IMG_W = 128, 128

AUDIO_DIR = "ICBHI_final_database"  # Folder with wav files

mel_tf = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=1024, win_length=WIN, hop_length=HOP,
    f_min=0.0, f_max=TARGET_SR//2, n_mels=N_MELS, power=2.0, center=True, window_fn=torch.hann_window
)
to_db = T.AmplitudeToDB(top_db=80.0)

def load_resample(path, sr=TARGET_SR):
    wav, s = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = T.Resample(s, sr)(wav)
    wav = wav.float()
    return wav

def wav_to_logmel_img(filename):
    path = os.path.join(AUDIO_DIR, filename)
    assert os.path.isfile(path), f"File not found: {path}"
    wav = load_resample(path, TARGET_SR)
    mel = mel_tf(wav)  # (n_mels, time) or (1, n_mels, time)
    logmel = to_db(mel)
    if len(logmel.shape) == 3:
        logmel = logmel.squeeze(0)
    mu, sigma = logmel.mean(), logmel.std().clamp_min(1e-6)
    logmel = (logmel - mu) / sigma
    img = torch.nn.functional.interpolate(
        logmel.unsqueeze(0).unsqueeze(0),
        size=(IMG_H, IMG_W),
        mode="bilinear",
        align_corners=False
    )
    return img.squeeze(0)  # (1, IMG_H, IMG_W)

# Read CSV
df = pd.read_csv("icbhi.csv")

# Create mel folder if missing
os.makedirs("mel", exist_ok=True)

failed_files = []

# Process and save mel spectrograms for all samples in CSV
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["filename"]
    try:
        mel_tensor = wav_to_logmel_img(filename)
        save_path = os.path.join("mel", f"{os.path.splitext(filename)[0]}_mel.pt")
        torch.save(mel_tensor, save_path)
    except Exception as e:
        print(f"Failed processing {filename}: {e}")
        failed_files.append(filename)

print(f"Processing complete! Failed files count: {len(failed_files)}")
if failed_files:
    print("Failed files:")
    for f in failed_files:
        print(f)
