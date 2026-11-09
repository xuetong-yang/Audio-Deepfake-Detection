import torch
import torch.nn as nn


class ConvAudioModel(nn.Module):
    """
    1D CNN over TIME.

    Data shape:
      - single sample: [D, T] = [180, 321]
      - batch:         [B, D, T] = [B, 180, 321]

    Conv1d expects input as [B, C, L]:
      - C = channels = D = 180 (feature dimension)
      - L = length   = T = 321 (time frames)

    Output:
      - logits: [B]  (raw scores, suitable for BCEWithLogitsLoss)
    """
    def __init__(self, feature_dim: int = 180):
        super().__init__()

        # Convolution blocks extract local patterns along time
        self.conv1 = nn.Sequential( 
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # L -> L/2, 321 -> 160
        )
        # padding=1 keeps the length for kernel_size=3; pooling then halves it
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 160 -> 80
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 80 -> 40
        )
        # output shape after conv3: [B, 256, 40]

        # Global average pooling: average each channel over time (length -> 1)
        # output shape after gap: [B, 256, 1]
        # The result works as an embedding of the whole utterance
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

        # The head maps the embedding to one score
        self.head = nn.Sequential(
            nn.Linear(256, 128), # linear 256 -> 128
            nn.ReLU(),
            nn.Dropout(0.3), # randomly zero some activations to prevent overfitting
            nn.Linear(128, 1), # linear 128 -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T] = [B, 180, 321]
        """
        x = self.conv1(x)               # [B, 64,  T/2]
        x = self.conv2(x)               # [B, 128, T/4]
        x = self.conv3(x)               # [B, 256, T/8]
        x = self.gap(x).squeeze(-1)     # [B, 256]. squeeze(-1) means remove last dim if it's 1
        logits = self.head(x).squeeze(-1)  # [B]
        return logits


class RNNAudioModel(nn.Module):
    """
    BiLSTM over TIME.

    Dataset provides batch as [B, D, T] = [B, 180, 321].
    LSTM with batch_first=True expects [B, T, D], so we transpose.

    Output:
      - logits: [B] (raw scores)
    """
    def __init__(self, feature_dim: int = 180, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,     # D=180
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,           # expects [B, T, D]
            bidirectional=True,         # output dim = 2 * hidden_dim = 256
            dropout=0.3 if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T] -> transpose -> [B, T, D]
        """
        x = x.transpose(1, 2)    # [B, T, D] = [B, 321, 180]
        out, _ = self.lstm(x)    # [B, T, 2H]: output size depends on hidden_size, not on D
        h = out.mean(dim=1)      # [B, 256]  (mean over time)mean pooling
        logits = self.head(h).squeeze(-1)  # [B]
        return logits
