# lstm_inference.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- Model Definition (must match train_lstm.py) ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 256, batch_first=True, bidirectional=False, dropout=0, num_layers=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.drop1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=False, dropout=0, num_layers=1)
        self.bn2 = nn.BatchNorm1d(1)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.bn1(out)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.bn2(out)
        out = self.drop2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)

# --- Load tokenizer and Electra model ---
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/araelectra-base-discriminator")
electra = AutoModel.from_pretrained("aubmindlab/araelectra-base-discriminator")
electra.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = electra(**inputs)
        return outputs.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()

# --- Inference function ---
def predict(text, model, device):
    emb = get_embedding(text)
    x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, hidden_size)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
    return pred
