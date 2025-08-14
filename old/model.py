"""
Provides a minimal convolutional neural network for MNIST classification.
"""

from torch import nn


class SmallCNN(nn.Module):
    """Three-layer CNN for MNIST (1x28x28 -> 10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)  # halves H & W (to 14x14)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
