import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from speechbrain.inference import EncoderClassifier
import torchaudio

SAMPLE_RATE = 16000
WAKEWORD_DURATION = 3
COMMAND_DURATION = 3

class MultiClassModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2_label = torch.nn.Linear(128, 3)
        self.fc2_auth = torch.nn.Linear(128, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2_label(x), self.fc2_auth(x)

def pad_or_truncate(waveform, target_len):
    if waveform.shape[1] > target_len:
        return waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
        pad_width = target_len - waveform.shape[1]
        return torch.nn.functional.pad(waveform, (0, pad_width))
    else:
        return waveform

def listen_and_predict(model, ecapa, device, duration, input_dim):
    print("Listening...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                        channels=1, dtype='float32')
    sd.wait()
    waveform = torch.tensor(audio_data.T, dtype=torch.float32)
    waveform = pad_or_truncate(waveform, SAMPLE_RATE * duration).to(device)
    with torch.no_grad():
        embedding = ecapa.encode_batch(waveform).flatten().unsqueeze(0)
        label_output, auth_output = model(embedding)
        label_probs = torch.softmax(label_output, dim=1).cpu().detach().numpy().flatten()
        auth_probs = torch.softmax(auth_output, dim=1).cpu().detach().numpy().flatten()
        label_pred = np.argmax(label_probs)
        auth_pred = np.argmax(auth_probs)
    return label_pred, auth_pred, label_probs, auth_probs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
    )
    dummy_waveform = torch.zeros(1, SAMPLE_RATE).to(device)
    with torch.no_grad():
        input_dim = ecapa.encode_batch(dummy_waveform).flatten().shape[0]
    model = MultiClassModel(input_dim).to(device)
    model.load_state_dict(torch.load("multi_class_model.pt", map_location=device, weights_only=True))
    model.eval()

    print("Say your wake word...") # Alexa
    label_pred, auth_pred, label_probs, auth_probs = listen_and_predict(
        model, ecapa, device, WAKEWORD_DURATION, input_dim
    )

    auth_prob = auth_probs[1]
    command_prob = label_probs[1]
    threshold = 0.4
    #if label_pred == 0 and auth_pred == 1:
    if label_pred == 0 and auth_prob > 1e-5:
        print("System Activated!")
        print("Say your command...")
        label_pred, auth_pred, label_probs, auth_probs = listen_and_predict(
            model, ecapa, device, COMMAND_DURATION, input_dim
        )
        auth_prob = auth_probs[1]
        command_prob = label_probs[1]
        if command_prob > 1e-6 and auth_prob > 1e-7:
            # TODO: Add action logic for multiple commands (depending on device)
            print("🎶 Playing Northern Attitude by Noah Kahan🎶")
        else: 
            print("Intruder!")

    elif auth_pred < 1e-3:
        print("Intruder!")
    else:
        print("No wake word detected.")
