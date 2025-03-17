import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from lion_pytorch import Lion
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_float32_matmul_precision('high')  

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
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


def load_data(cache_tensors=True):
    # Cache dataset as tensors for faster access"
    cache_file = "processed_data_tensors.pt"
    
    if cache_tensors and os.path.exists(cache_file):
        data_tensor = torch.load(cache_file)
        dataset = TensorDataset(data_tensor)
        return dataset
    
    subset_1 = np.load("subset_1.npy", mmap_mode='r') 
    subset_2 = np.load("subset_2.npy", mmap_mode='r')
    subset_3 = np.load("subset_3.npy", mmap_mode='r')
    
    batch_size = 1000
    all_tensors = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((152, 224), antialias=True),
    ])
    
    total_samples = len(subset_1) + len(subset_2) + len(subset_3)
    processed = 0
    
    # Process subset_1
    for i in range(0, len(subset_1), batch_size):
        batch = subset_1[i:i+batch_size].reshape(-1, 150, 225, 3)
        tensors = torch.stack([transform(img) for img in batch])
        all_tensors.append(tensors)
        processed += len(batch)
        print(f"Processed {processed}/{total_samples} images")
    
    # Process subset_2
    for i in range(0, len(subset_2), batch_size):
        batch = subset_2[i:i+batch_size].reshape(-1, 150, 225, 3)
        tensors = torch.stack([transform(img) for img in batch])
        all_tensors.append(tensors)
        processed += len(batch)
        print(f"Processed {processed}/{total_samples} images")
    
    # Process subset_3
    for i in range(0, len(subset_3), batch_size):
        batch = subset_3[i:i+batch_size].reshape(-1, 150, 225, 3)
        tensors = torch.stack([transform(img) for img in batch])
        all_tensors.append(tensors)
        processed += len(batch)
        print(f"Processed {processed}/{total_samples} images")
    
    data_tensor = torch.cat(all_tensors)
    
    if cache_tensors:
        torch.save(data_tensor, cache_file)
    
    dataset = TensorDataset(data_tensor)
    return dataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefetch_factor = 2
    num_workers = min(6, os.cpu_count() or 4)
    
    dataset = load_data(cache_tensors=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=128,  
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    model = Autoencoder().to(device, memory_format=torch.channels_last)
    
    # Print model parameters and compression ratio
    input_size = 3 * 152 * 224
    encoder_output = model.encoder(torch.zeros(1, 3, 152, 224).to(device, memory_format=torch.channels_last))
    encoded_size = encoder_output.numel()
    compression_ratio = input_size / encoded_size
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = HybridLoss(device, alpha=0.7)
    optimizer = Lion(model.parameters(), lr=0.002, weight_decay=0.005)  
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart every 5 epochs
        T_mult=1,
        eta_min=1e-5,
    ) 
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None
    
    best_loss_file = "best_loss.txt"
    best_loss = float('inf')
    if os.path.exists(best_loss_file):
        with open(best_loss_file, "r") as f:
            try:
                best_loss = float(f.read().strip())
                print(f"Previous best loss: {best_loss:.4f}")
            except ValueError:
                pass

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        running_loss = 0
        log_interval = 10  # Display progress every 10 batches
        
        for batch_idx, batch in enumerate(dataloader):
            img = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True) 
            
            with torch.amp.autocast(device_type=device.type) if use_amp else torch.enable_grad():
                recon = model(img)
                loss = criterion(recon, img)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            running_loss += current_loss
            
            # Display progress
            if (batch_idx + 1) % log_interval == 0:
                avg_running_loss = running_loss / log_interval
                running_loss = 0
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {avg_running_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model (with a new filename to avoid conflicts)
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(model.state_dict(), 'autoencoder_new.pth')
                with open(best_loss_file, "w") as f:
                    f.write(str(best_loss))
                print(f"New best model saved with loss: {best_loss:.4f}")
                
        # Report epoch stats
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Current LR: {optimizer.param_groups[0]["lr"]:.6f}')


if __name__ == '__main__':
    train()
