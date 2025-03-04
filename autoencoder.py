import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

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
    

transform = transforms.Compose([transforms.ToTensor()])
images = np.load("subset_1.npy")
dataloader = torch.utils.data.DataLoader(images, batch_size=64, shuffle=True)

model = Autoencoder()
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
outputs = []
for epoch in range(num_epochs):
    for img in dataloader:
        recon = model(img)
        loss = criterion(recon, img)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    outputs.append((epoch, img, recon))