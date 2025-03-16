import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure

torch.backends.cudnn.benchmark = True

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, groups=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=1),
            nn.ReLU(),
        )

        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class HybridLoss(nn.Module):
    def __init__(self, device, alpha=0.8):
        super(HybridLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        ssim_loss = 1 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


def load_data():
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
    return dataset

def train():
    dataset = load_data()
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, 
        num_workers=os.cpu_count() // 2,  
        pin_memory=True, persistent_workers=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device, memory_format=torch.channels_last)
    criterion = HybridLoss(device)  
    optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.95, 0.98))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_loss_file = "best_loss.txt"
    best_loss = float('inf')
    if os.path.exists(best_loss_file):
        with open(best_loss_file, "r") as f:
            try:
                best_loss = float(f.read().strip())
            except ValueError:
                pass  

    input_size = 3 * 152 * 224  
    encoded_size = 32 * 19 * 28  
    compression_ratio = input_size / encoded_size
    print(f"Compression Ratio: {compression_ratio:.2f}")

    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            img = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.enable_grad():
                recon = model(img)
                loss = criterion(recon, img)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'autoencoder.pth')
                with open(best_loss_file, "w") as f:
                    f.write(str(best_loss))  
                print(f"New all-time best model saved with loss: {best_loss:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)  # Use avg_loss here, not loss.item()
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


if __name__ == '__main__':
    train()
