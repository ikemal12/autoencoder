import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

images = np.load("subset_1.npy")
print(images.shape)

i = 201
print(plt.imshow(np.reshape(images[i, :], (150, 225, 3))))

class Autoencoder(nn.Module):
    def __init__(self):
        
        self.encoder = nn.Sequential(
            nn.Linear(150*225, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 150*225),
            nn.Sigmoid()
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decode(encoded)
        return decoded