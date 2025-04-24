import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from speechbrain.inference import EncoderClassifier
import torchaudio

SAMPLE_RATE = 16000
AUDIO_LEN = 3  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10

# --- Model Definition ---
class MultiClassModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Word type ("label")
        self.fc2_label = nn.Linear(128, 3)
        # User authentication ("auth")
        self.fc2_auth = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2_label(x), self.fc2_auth(x)

# --- Dataset ---
class AudioDataset(Dataset):
    def __init__(self, csv_path, ecapa_model, sample_rate=SAMPLE_RATE, audio_len=AUDIO_LEN):
        self.df = pd.read_csv(csv_path)
        self.ecapa = ecapa_model
        self.sample_rate = sample_rate
        self.target_len = sample_rate * audio_len

    def __len__(self):
        return len(self.df)

    def pad_or_truncate(self, waveform):
        if waveform.shape[1] > self.target_len:
            return waveform[:, :self.target_len]
        elif waveform.shape[1] < self.target_len:
            pad_width = self.target_len - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, pad_width))
        else:
            return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label = int(row['label'])
        auth = int(row['auth'])
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = self.pad_or_truncate(waveform)
        with torch.no_grad():
            embedding = self.ecapa.encode_batch(waveform).flatten()
        return embedding, label, auth

# --- Class Weights ---
def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1. / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# --- Training Loop ---
def train(model, loader, optimizer, criterion_label, criterion_auth):
    model.train()
    total_loss = 0
    correct_label = 0
    correct_auth = 0
    total = 0
    for x, y_label, y_auth in loader:
        x = x.to(DEVICE)
        y_label = y_label.to(DEVICE)
        y_auth = y_auth.to(DEVICE)
        optimizer.zero_grad()
        out_label, out_auth = model(x)
        loss_label = criterion_label(out_label, y_label)
        loss_auth = criterion_auth(out_auth, y_auth)
        loss = loss_label + loss_auth
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Accuracy
        pred_label = torch.argmax(out_label, dim=1)
        pred_auth = torch.argmax(out_auth, dim=1)
        correct_label += (pred_label == y_label).sum().item()
        correct_auth += (pred_auth == y_auth).sum().item()
        total += y_label.size(0)
    avg_loss = total_loss / len(loader)
    label_acc = correct_label / total
    auth_acc = correct_auth / total
    return avg_loss, label_acc, auth_acc

def evaluate(model, loader, criterion_label, criterion_auth):
    model.eval()
    total_loss = 0
    correct_label = 0
    correct_auth = 0
    total = 0
    with torch.no_grad():
        for x, y_label, y_auth in loader:
            x = x.to(DEVICE)
            y_label = y_label.to(DEVICE)
            y_auth = y_auth.to(DEVICE)
            out_label, out_auth = model(x)
            loss_label = criterion_label(out_label, y_label)
            loss_auth = criterion_auth(out_auth, y_auth)
            loss = loss_label + loss_auth
            total_loss += loss.item()
            pred_label = torch.argmax(out_label, dim=1)
            pred_auth = torch.argmax(out_auth, dim=1)
            correct_label += (pred_label == y_label).sum().item()
            correct_auth += (pred_auth == y_auth).sum().item()
            total += y_label.size(0)
    avg_loss = total_loss / len(loader)
    label_acc = correct_label / total
    auth_acc = correct_auth / total
    return avg_loss, label_acc, auth_acc

# --- Main ---
if __name__ == "__main__":
    # Load ECAPA model for embeddings
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE}
    )

    # Prepare dataset
    train_csv = "train.csv"  
    val_csv = "val.csv" 

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Compute class weights for label and auth
    label_weights = compute_class_weights(train_df['label'].values, 3)
    auth_weights = compute_class_weights(train_df['auth'].values, 2)

    # Datasets and loaders
    train_set = AudioDataset(train_csv, ecapa)
    val_set = AudioDataset(val_csv, ecapa)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Model
    dummy_waveform = torch.zeros(1, SAMPLE_RATE).to(DEVICE)
    with torch.no_grad():
        dummy_embedding = ecapa.encode_batch(dummy_waveform).flatten()
        input_dim = dummy_embedding.shape[0]
    model = MultiClassModel(input_dim).to(DEVICE)

    # Loss and optimizer
    criterion_label = nn.CrossEntropyLoss(weight=label_weights)
    criterion_auth = nn.CrossEntropyLoss(weight=auth_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss, train_label_acc, train_auth_acc = train(
            model, train_loader, optimizer, criterion_label, criterion_auth
        )
        val_loss, val_label_acc, val_auth_acc = evaluate(
            model, val_loader, criterion_label, criterion_auth
        )
        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} | Train Label Acc: {train_label_acc:.3f} | Train Auth Acc: {train_auth_acc:.3f} "
            f"| Val Loss: {val_loss:.4f} | Val Label Acc: {val_label_acc:.3f} | Val Auth Acc: {val_auth_acc:.3f}"
        )

    # Save model
    torch.save(model.state_dict(), "multi_class_model.pt")
    print("Model saved as multi_class_model.pt")
