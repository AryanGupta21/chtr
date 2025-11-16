import os
import glob
import numpy as np
import librosa
import scipy.signal as sps
import soundfile as sf

in_dir = "ICBHI_final_database"
out_dir = "ICBHI_lung_clean"
os.makedirs(out_dir, exist_ok=True)

n_fft = 1024
hop = n_fft // 4
win = sps.windows.hann(n_fft, sym=False)

def medfilt_time(x, k=9):
    pad = k // 2
    xpad = np.pad(x, ((0,0),(pad,pad)), mode="edge")
    out = np.empty_like(x)
    for t in range(x.shape[1]):
        out[:, t] = np.median(xpad[:, t:t+k], axis=1)
    return out

wav_paths = glob.glob(os.path.join(in_dir, "**", "*.wav"), recursive=True)
print("Found", len(wav_paths), "wav files")

for wav_path in wav_paths:
    print("Processing:", wav_path)
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win)
    mag, phase = np.abs(S), np.exp(1j * np.angle(S))

    freqs = np.linspace(0, sr/2, mag.shape[0])
    lung_band = ((freqs >= 200) & (freqs <= 1000)).astype(float)
    heart_band = ((freqs >= 50) & (freqs <= 200)).astype(float)

    E_lung = (mag * lung_band[:, None])**2
    E_heart = (mag * heart_band[:, None])**2
    E_lung_s = medfilt_time(E_lung, k=11)
    E_heart_s = medfilt_time(E_heart, k=11)

    eps = 1e-8
    mask = (E_lung_s + eps) / (E_lung_s + E_heart_s + eps)
    mask = np.clip(mask, 0.0, 1.0)

    S_lung = mask * mag * phase
    y_lung = librosa.istft(S_lung, hop_length=hop, window=win)

    base = os.path.basename(wav_path)
    out_path = os.path.join(out_dir, base)

    # REPLACE THIS:
    # librosa.output.write_wav(out_path, y_lung, sr)

    # WITH THIS:
    sf.write(out_path, y_lung, sr)   # float32/float64 is fine
    print("Saved:", out_path)
print("All done.")