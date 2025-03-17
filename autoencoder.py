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
    def __init__(self, latent_dim=128, use_attention=True):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),  

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),

            nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        if self.use_attention:
            encoded = self.attention(encoded)
        decoded = self.decoder(encoded)
        return decoded
    

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class HybridLoss(nn.Module):
    def __init__(self, device, alpha=0.8, beta=0.1, gamma=0.2):
        super(HybridLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target, perceptual_loss=None):
        ssim_loss = 1 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)

        if perceptual_loss is not None:
            return self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * perceptual_loss
        else:
            return (self.alpha / (self.alpha + self.beta)) * mse_loss + (self.beta / (self.alpha + self.beta)) * ssim_loss


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
    
    process_subset(subset_1, batch_size, all_tensors, transform, total_samples)
    process_subset(subset_2, batch_size, all_tensors, transform, total_samples)
    process_subset(subset_3, batch_size, all_tensors, transform, total_samples)
    
    data_tensor = torch.cat(all_tensors)
    
    if cache_tensors:
        torch.save(data_tensor, cache_file)
    
    dataset = TensorDataset(data_tensor)
    return dataset


def process_subset(subset, batch_size, all_tensors, transform, total_samples):
    processed = 0
    for i in range(0, len(subset), batch_size):
        batch = subset[i:i+batch_size].reshape(-1, 150, 225, 3)
        tensors = torch.stack([transform(img) for img in batch])
        all_tensors.append(tensors)
        processed += len(batch)
        print(f"Processed {processed}/{total_samples} images")


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefetch_factor = 4
    num_workers = min(8, os.cpu_count() or 4)
    
    dataset = load_data(cache_tensors=True)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    model = Autoencoder(latent_dim=96, use_attention=True).to(device, memory_format=torch.channels_last)
    
    # Print model parameters and compression ratio
    input_size = 3 * 152 * 224
    encoder_output = model.encoder(torch.zeros(1, 3, 152, 224).to(device, memory_format=torch.channels_last))
    encoded_size = encoder_output.numel()
    compression_ratio = input_size / encoded_size
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = HybridLoss(device, alpha=0.7, beta=0.2, gamma=0.1)
    optimizer = Lion(model.parameters(), lr=0.0025, weight_decay=0.003)  
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=25,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000
    )
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    best_loss_file = "best_loss.txt"
    checkpoint_file = "autoencoder_checkpoint.pth"
    best_loss = float('inf')
    start_epoch = 0

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
    elif os.path.exists(best_loss_file):
        with open(best_loss_file, "r") as f:
            try:
                best_loss = float(f.read().strip())
                print(f"Previous best loss: {best_loss:.4f}")
            except ValueError:
                pass

    # Training loop
    num_epochs = 25
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        running_loss = 0
        log_interval = 10  # Display progress every 10 batches
        
        for batch_idx, batch in enumerate(train_loader):
            img = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True) 
            
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recon = model(img)
                loss = criterion(recon, img)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            running_loss += current_loss
            
            if (batch_idx + 1) % log_interval == 0:
                avg_running_loss = running_loss / log_interval
                running_loss = 0
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_running_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model periodically
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, "autoencoder_checkpoint.pth")

        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                img = batch[0].to(device, non_blocking=True, memory_format=torch.channels_last)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    recon = model(img)
                    loss = criterion(recon, img)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Report epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, 'autoencoder_best.pth')
            with open(best_loss_file, "w") as f:
                f.write(str(best_loss))
            print(f"New best model saved with loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

if __name__ == '__main__':
    train()
