import torch.nn as nn


class SmallCNN(nn.Module):
    """Two conv-ReLU-max-pool blocks and a two-layer classifier with dropout 0.1 (about 421k parameters)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 2),
        )

    def forward(self, x):
        z = self.conv(x)
        return self.fc(z.view(z.size(0), -1))
