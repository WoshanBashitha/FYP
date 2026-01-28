import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bi-LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because bidirectional

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden*2)
        out = out[:, -1, :]  # take output of last time step
        out = self.fc(out)
        return out

# -----------------------------
# Training function
# -----------------------------
def train_model(model, X_train, y_train, epochs=50, batch_size=32, lr=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                             torch.tensor(y_train, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs.squeeze(), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.6f}")
    return model

# -----------------------------
# Prediction function
# -----------------------------
def predict_model(model, X, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).cpu().numpy()
    return preds