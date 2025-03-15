import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 76, 112)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 38, 56)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 19, 28)
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 38, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 76, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (3, 152, 224)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))

# loading data 
subset_1 = np.load("subset_1.npy")
subset_2 = np.load("subset_2.npy")
subset_3 = np.load("subset_3.npy")
data = np.concatenate((subset_1, subset_2, subset_3), axis=0)
data = data.reshape(-1, 150, 225, 3)  

transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((152, 224)),  
    transforms.ToTensor()  
])

data_tensor = torch.stack([transform(img) for img in data])
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cpu')
model = Autoencoder().to(device)
criterion = RMSELoss() #nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cuda')
best_loss = float('inf')  

# training loop
num_epochs = 20
for epoch in range(num_epochs):
    for batch in dataloader:
        img = batch[0].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            recon = model(img)
            loss = criterion(recon, img)

        scaler.scale(loss).backward()  
        scaler.step(optimizer)
        scaler.update() 

        # save the model only if the loss improves
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'autoencoder.pth')
            print(f"Best model saved with loss: {best_loss:.4f}")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
