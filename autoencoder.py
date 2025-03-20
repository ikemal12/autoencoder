import torch
import os
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from lion_pytorch import Lion
from torch.cuda.amp import autocast, GradScaler
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Memory management functions
def free_memory():
    """Free up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def parse_args():
    parser = argparse.ArgumentParser(description="Train an autoencoder for image compression.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model from scratch, ignoring checkpoints.")
    return parser.parse_args()

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=96, use_attention=True):
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
        self.channel_dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        encoded = self.encoder(x)
        if self.use_attention:
            encoded = self.attention(encoded)
        if self.training:
            encoded = self.channel_dropout(encoded)
        decoded = self.decoder(encoded)
        return decoded, encoded

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
    def __init__(self, device, alpha=0.6, beta=0.35, gamma=0.05):
        super(HybridLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target, encoded=None):
        ssim_loss = 1 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)
        if encoded is not None:
            latent_loss = torch.mean(torch.abs(encoded))
            return self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * latent_loss
        else:
            return self.alpha * mse_loss + self.beta * ssim_loss

def load_data(cache_tensors=True):
    cache_file = "processed_data_tensors.pt"
    if cache_tensors and os.path.exists(cache_file):
        data_tensor = torch.load(cache_file)
        dataset = TensorDataset(data_tensor)
        return dataset
    subset_1 = np.load("subset_1.npy", mmap_mode='r')
    subset_2 = np.load("subset_2.npy", mmap_mode='r')
    subset_3 = np.load("subset_3.npy", mmap_mode='r')
    batch_size = 500
    all_tensors = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((152, 224), antialias=True),
    ])
    total_samples = len(subset_1) + len(subset_2) + len(subset_3)
    process_subset(subset_1, batch_size, all_tensors, transform, total_samples)
    free_memory()
    process_subset(subset_2, batch_size, all_tensors, transform, total_samples)
    free_memory()
    process_subset(subset_3, batch_size, all_tensors, transform, total_samples)
    free_memory()
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
        logging.info(f"Processed {processed}/{total_samples} images")

def visualize_results(model, val_loader, device, epoch, num_images=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of images from the validation loader
        for batch in val_loader:
            images = batch[0].to(device)
            break  # We only need one batch

        # Select the first `num_images` images from the batch
        images = images[:num_images]

        # Pass the images through the autoencoder
        decompressed_images, compressed_images = model(images)

        # Convert tensors to numpy arrays and permute dimensions for visualization
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        decompressed_images = decompressed_images.cpu().numpy().transpose(0, 2, 3, 1)

        # Display the images
        plt.figure(figsize=(15, 5 * num_images))
        for i in range(num_images):
            # Original Image
            plt.subplot(num_images, 3, 3 * i + 1)
            plt.imshow(images[i])
            plt.title("Original Image")
            plt.axis('off')

            # Compressed Image (Latent Representation)
            plt.subplot(num_images, 3, 3 * i + 2)
            latent_image = compressed_images[i].cpu().numpy()
            plt.imshow(latent_image.mean(0), cmap='viridis')  # Take the mean across channels
            plt.title("Compressed Image (Latent Space)")
            plt.axis('off')

            # Decompressed Image
            plt.subplot(num_images, 3, 3 * i + 3)
            plt.imshow(decompressed_images[i])
            plt.title(f"Decompressed Image (Epoch {epoch + 1})")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def train(retrain=False): 
    free_memory()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = min(4, os.cpu_count() or 2)
    prefetch_factor = 2
    logging.info("Loading data...")
    dataset = load_data(cache_tensors=True)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    batch_size = 128
    accumulation_steps = 2
    logging.info(f"Creating data loaders with batch size {batch_size}...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    logging.info("Creating model...")
    latent_dim = 96
    model = Autoencoder(latent_dim=latent_dim, use_attention=True).to(device)
    input_size = 3 * 152 * 224
    with torch.no_grad():
        _, encoded = model(torch.zeros(1, 3, 152, 224).to(device))
        encoded_size = encoded.numel()
        compression_ratio = input_size / encoded_size
        effective_compression = compression_ratio * 2
        logging.info(f"Compression Ratio: {compression_ratio:.2f}")
        logging.info(f"Effective Compression Ratio (with dropout): {effective_compression:.2f}")
        logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    criterion = HybridLoss(device, alpha=0.6, beta=0.35, gamma=0.05)
    optimizer = Lion(model.parameters(), lr=0.002, weight_decay=0.0025)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=20,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_loss_file = "best_loss.txt"
    checkpoint_file = "autoencoder_checkpoint.pth"
    best_loss = float('inf')
    start_epoch = 0

    # Load checkpoint only if not retraining
    if not retrain and os.path.exists(checkpoint_file):
        logging.info("Loading checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            best_loss = checkpoint['best_loss']
            logging.info(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Starting fresh training")
    elif retrain:
        logging.info("Retraining from scratch (ignoring checkpoint).")
    elif os.path.exists(best_loss_file):
        with open(best_loss_file, "r") as f:
            try:
                best_loss = float(f.read().strip())
                logging.info(f"Previous best loss: {best_loss:.4f}")
            except ValueError:
                pass

    num_epochs = 20
    patience = 5
    patience_counter = 0
    total_start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logging.info(f"\nStarting epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        running_loss = 0
        log_interval = 10
        batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            img = batch[0].to(device, non_blocking=True)
            if batch_idx % 200 == 0:
                free_memory()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recon, encoded = model(img)
                loss = criterion(recon, img, encoded)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                current_loss = loss.item() * accumulation_steps
                epoch_loss += current_loss
                running_loss += current_loss
                batch_count += 1
                if (batch_idx + 1) % (log_interval * accumulation_steps) == 0:
                    avg_running_loss = running_loss / log_interval
                    running_loss = 0
                    logging.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                                 f"Loss: {avg_running_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
        }, checkpoint_file)
        free_memory()
        model.eval()
        val_loss = 0
        val_ssim = 0
        val_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                img = batch[0].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    recon, encoded = model(img)
                    loss = criterion(recon, img, encoded)
                    ssim_value = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(recon, img)
                val_loss += loss.item()
                val_ssim += ssim_value.item()
                val_count += 1
                if batch_idx % 50 == 0:
                    free_memory()
        val_loss /= val_count
        val_ssim /= val_count
        avg_train_loss = epoch_loss / batch_count
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                     f'Val SSIM: {val_ssim:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}, Time: {epoch_duration:.2f} seconds')
        
        # Visualize results after every epoch
        visualize_results(model, val_loader, device, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'compression_ratio': compression_ratio,
                'val_ssim': val_ssim,
            }, 'autoencoder_best.pth')
            with open(best_loss_file, "w") as f:
                f.write(str(best_loss))
            logging.info(f"New best model saved with loss: {best_loss:.4f}, SSIM: {val_ssim:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        free_memory()
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total training time: {total_duration:.2f} seconds")

if __name__ == '__main__':
    args = parse_args()  
    try:
        train(retrain=args.retrain)  # Pass retrain argument to train function
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()