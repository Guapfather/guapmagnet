import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# CONFIG
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
SEQ_LEN = 60  # Use last 60 candles
MODEL_SAVE_PATH = "guapmagnet_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class ForexDataset(Dataset):
    def __init__(self, df):
        self.X = []
        self.y = []
        for i in range(SEQ_LEN, len(df)):
            features = df.iloc[i-SEQ_LEN:i][['open', 'high', 'low', 'close', 'volume']].values
            label = 1 if df.iloc[i]['close'] > df.iloc[i]['open'] else 0  # BUY if green candle, else SELL
            self.X.append(features)
            self.y.append(label)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model
class GuapMagnetBrain(nn.Module):
    def __init__(self):
        super(GuapMagnetBrain, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)  # Outputs: BUY / SELL

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = h_lstm[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def train_model(train_loader):
    model = GuapMagnetBrain().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved as {MODEL_SAVE_PATH}")
    return model

if __name__ == "__main__":
    df = pd.read_csv("guapmagnet_dataset.csv")
    dataset = ForexDataset(df)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = train_model(train_loader)
