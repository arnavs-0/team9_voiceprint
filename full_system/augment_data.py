import os
import torchaudio
import torch
import random

AUGMENTED_DIR = "dataset/augmented_user_command"
os.makedirs(AUGMENTED_DIR, exist_ok=True)

def augment_and_save(input_path, output_path, sample_rate=16000):
    waveform, sr = torchaudio.load(input_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    n_steps = random.uniform(-2, 2)
    pitch_shift = torchaudio.transforms.PitchShift(sample_rate, n_steps=n_steps)
    waveform = pitch_shift(waveform)
    noise = torch.randn_like(waveform) * 0.005 * random.uniform(0.5, 1.5)
    waveform = waveform + noise
    torchaudio.save(output_path, waveform.detach(), sample_rate)

def augment_folder(folder, n_augments=10):
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    for f in files:
        input_path = os.path.join(folder, f)
        for i in range(n_augments):
            out_name = f"{os.path.splitext(f)[0]}_aug{i}.wav"
            output_path = os.path.join(AUGMENTED_DIR, out_name)
            augment_and_save(input_path, output_path)

if __name__ == "__main__":
    augment_folder("dataset/user_command", n_augments=100)
