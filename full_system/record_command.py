import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

SAMPLE_RATE = 16000
DURATION = 3  

def record_command(save_folder, filename):
    os.makedirs(save_folder, exist_ok=True)
    print("Get ready to record your command.")
    input("Press Enter to start recording...")
    print("Recording...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    filepath = os.path.join(save_folder, filename)
    wav.write(filepath, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Saved to {filepath}")

if __name__ == "__main__":
    command_text = input("Type the command you want to add (e.g., 'turn on the lights'): ").strip().replace(" ", "_")
    save_folder = os.path.join("dataset", "user_command")
    #TODO: Add logic for multiple users 
    filename = f"{command_text}_user.wav"
    record_command(save_folder, filename)
