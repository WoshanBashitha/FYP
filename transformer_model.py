import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Input projection (features → d_model)
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding (simple learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 500, d_model))  # max seq len = 500

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Transformer encoder
        x = self.transformer(x)

        # Take last time step
        x = x[:, -1, :]

        # Output
        x = self.fc(x)

        return x


# -----------------------------
# Prediction function (same style as BiLSTM)
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