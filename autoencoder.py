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
import optuna
from optuna.trial import Trial
import joblib
import gc
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# Define fixed choices for categorical parameters
BATCH_SIZE_CHOICES = [16, 32, 64]  # Fixed batch size choices
OPTIMIZER_CHOICES = ['Lion', 'AdamW']  # Fixed optimizer choices

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
    logging.info(f"Memory usage: {memory_usage:.2f} MB")

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
    def __init__(self, latent_dim=96, use_attention=True, 
                 dropout_rate=0.5, residual_layers=2, filters=(32, 64, 128)):
        super(Autoencoder, self).__init__()
        
        # Create encoder layers
        encoder_layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(filters):
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            # Add residual blocks based on parameter
            if i < residual_layers:
                encoder_layers.append(ResidualBlock(out_channels))
                
            in_channels = out_channels
        
        # Final conv layer to latent space
        encoder_layers.extend([
            nn.Conv2d(filters[-1], latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder layers
        decoder_layers = []
        in_channels = latent_dim
        reversed_filters = list(reversed(filters))
        
        decoder_layers.extend([
            nn.ConvTranspose2d(latent_dim, reversed_filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(reversed_filters[0]),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        if 0 < residual_layers:
            decoder_layers.append(ResidualBlock(reversed_filters[0]))
        
        for i in range(len(reversed_filters)-1):
            decoder_layers.extend([
                nn.ConvTranspose2d(reversed_filters[i], reversed_filters[i+1], 
                                  kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(reversed_filters[i+1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if i+1 < residual_layers:
                decoder_layers.append(ResidualBlock(reversed_filters[i+1]))
        
        # Final output layer
        decoder_layers.extend([
            nn.ConvTranspose2d(reversed_filters[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(latent_dim)
        self.channel_dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        encoded = self.encoder(x)
        if self.use_attention:
            encoded = self.attention(encoded)
        if self.training:
            encoded = self.channel_dropout(encoded)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
def create_model_from_params(trial, device):
    # Hyperparameters to tune
    latent_dim = trial.suggest_int('latent_dim', 48, 196, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    residual_layers = trial.suggest_int('residual_layers', 1, 3)
    
    # Dynamically build filter configuration
    filters = []
    num_filters = trial.suggest_int('num_filters', 2, 4)
    for i in range(num_filters):
        filters.append(trial.suggest_int(f'filter_{i}', 16, 256, step=16))
    
    # Create model with these parameters
    model = Autoencoder(
        latent_dim=latent_dim, 
        use_attention=use_attention,
        dropout_rate=dropout_rate,
        residual_layers=residual_layers,
        filters=tuple(filters)
    ).to(device)
    
    return model

def objective(trial: Trial, train_dataset, val_dataset, device, num_epochs=5):
    try:
        # Create model with hyperparameters from trial
        model = create_model_from_params(trial, device)
        
        # Get hyperparameters for optimizer and loss
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        # Loss function hyperparameters
        alpha = trial.suggest_float('alpha', 0.3, 0.8)
        beta = trial.suggest_float('beta', 0.1, 0.6)
        gamma = trial.suggest_float('gamma', 0.01, 0.1)
        
        criterion = HybridLoss(device, alpha=alpha, beta=beta, gamma=gamma)
        
        # Choose optimizer from fixed choices
        optimizer_name = trial.suggest_categorical('optimizer', OPTIMIZER_CHOICES)
        if optimizer_name == 'Lion':
            optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Batch size from fixed choices
        batch_size = trial.suggest_categorical('batch_size', BATCH_SIZE_CHOICES)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Use OneCycleLR for all trials
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        )
        
        use_amp = torch.cuda.is_available()
        scaler = torch.amp.GradScaler(enabled=use_amp)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            accumulation_steps = 2  # Accumulate gradients over 2 batches

            for batch_idx, batch in enumerate(train_loader):
                img = batch[0].to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    recon, encoded = model(img)
                    loss = criterion(recon, img, encoded) / accumulation_steps  # Normalize loss
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                            
                epoch_loss += loss.item()
                batch_count += 1
                
                # Report intermediate values for pruning
                if batch_idx % 10 == 0:
                    trial.report(loss.item(), epoch * len(train_loader) + batch_idx)
                    
                    # Prune trial if it's not performing well
                    if trial.should_prune():
                        free_memory()
                        raise optuna.exceptions.TrialPruned()
            
            # Validation
            model.eval()
            val_loss = 0
            val_ssim = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    img = batch[0].to(device, non_blocking=True)
                    
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        recon, encoded = model(img)
                        loss = criterion(recon, img, encoded)
                        ssim_value = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(recon, img)
                    
                    val_loss += loss.item()
                    val_ssim += ssim_value.item()
                    val_count += 1
            
            val_loss /= val_count
            val_ssim /= val_count
            
            # Keep track of the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Report metrics to Optuna
            trial.report(val_loss, epoch)
            
            # Prune trial if it's not promising
            if trial.should_prune():
                free_memory()
                raise optuna.exceptions.TrialPruned()
        
        free_memory()
        return best_val_loss
    finally:
        # Free memory after each trial
        free_memory()
        gc.collect()

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
        # Make sure inputs have valid values to avoid NaN
        pred = torch.clamp(pred, 0.001, 0.999)  # Avoid extreme values for stability
        
        ssim_loss = 1 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)
        
        if encoded is not None:
            # Use L1 regularization with a small epsilon to avoid NaN
            latent_loss = torch.mean(torch.abs(encoded) + 1e-8)
            return self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * latent_loss
        else:
            return self.alpha * mse_loss + self.beta * ssim_loss
        
def visualize_optuna_results(study_name="autoencoder_hparam_study"):
    """Visualize the hyperparameter optimization results."""
    import optuna.visualization as vis
    import matplotlib.pyplot as plt
    
    # Load the study
    study = joblib.load(f"{study_name}.pkl")
    
    # Plot optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.show()
    
    # Plot parameter importances
    fig2 = vis.plot_param_importances(study)
    fig2.show()
    
    # Plot parallel coordinates
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.show()
    
    # Plot slice plot for specific parameters
    for param in ['latent_dim', 'dropout_rate', 'lr']:
        if param in study.best_params:
            fig = vis.plot_slice(study, params=[param])
            fig.show()

def load_data(cache_tensors=True):
    cache_file = "processed_data_tensors.pt"
    if cache_tensors and os.path.exists(cache_file):
        data_tensor = torch.load(cache_file)
        dataset = TensorDataset(data_tensor)
        return dataset

    # Load data in smaller chunks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((144, 224)),  # Ensure consistent dimensions
    ])

    all_tensors = []
    batch_size = 500  # Process data in smaller batches
    subsets = ["subset_1.npy", "subset_2.npy", "subset_3.npy"]
    total_samples = sum(len(np.load(subset, mmap_mode='r')) for subset in subsets)

    for subset in subsets:
        data = np.load(subset, mmap_mode='r')  # Use memory-mapped files
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].reshape(-1, 150, 225, 3)
            tensors = torch.stack([transform(img) for img in batch])
            all_tensors.append(tensors)
            logging.info(f"Processed {len(all_tensors) * batch_size}/{total_samples} images")
            free_memory()  # Free memory after each batch

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

        # Debugging: Check the range of original and decompressed images
        print("Original min:", images.min().item())
        print("Original max:", images.max().item())
        print("Decompressed min:", decompressed_images.min().item())
        print("Decompressed max:", decompressed_images.max().item())

        # Clamp decompressed images to [0, 1] range if necessary
        decompressed_images = torch.clamp(decompressed_images, 0, 1)

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
    
    # Load and prepare datasets
    logging.info("Loading data...")
    dataset = load_data(cache_tensors=True)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    log_memory_usage()
    
    # Begin with hyperparameter tuning if retrain is True or no prior study exists
    study_name = "autoencoder_hparam_study"
    storage_name = f"sqlite:///{study_name}.db"
    
    if retrain or not os.path.exists(f"{study_name}.db"):
        logging.info("Starting hyperparameter optimization...")
        
        # Create a new study or load existing one
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name, 
            direction="minimize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Set up the objective function
        objective_with_datasets = lambda trial: objective(
            trial, train_dataset, val_dataset, device, num_epochs=3
        )
        
        # Run optimization
        study.optimize(
            objective_with_datasets, 
            n_trials=10,  # Adjust based on your compute resources
            timeout=86400,  # 24 hours timeout
            gc_after_trial=True
        )
        
        # Save study results
        joblib.dump(study, f"{study_name}.pkl")
        
        # Print optimization results
        logging.info("Best trial:")
        trial = study.best_trial
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")
    else:
        # Load the existing study
        try:
            # Try to load from pickle file first
            study = joblib.load(f"{study_name}.pkl")
            logging.info("Loaded existing hyperparameter study from pickle file")
        except (FileNotFoundError, EOFError, Exception) as e:
            # If pkl doesn't exist or is corrupted, load from the database
            logging.warning(f"Failed to load study from pickle file: {e}. Loading from database instead.")
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            logging.info("Loaded existing hyperparameter study from database")
            # Save as pickle for next time
            joblib.dump(study, f"{study_name}.pkl")
        
        # Check if the study has trials before accessing best_value
        if len(study.trials) == 0:
            logging.warning("No trials found in the study. Starting fresh hyperparameter optimization.")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name, 
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )
            objective_with_datasets = lambda trial: objective(
                trial, train_dataset, val_dataset, device, num_epochs=3
            )
            study.optimize(
                objective_with_datasets, 
                n_trials=10,  # Adjust based on your compute resources
                timeout=86400,  # 24 hours timeout
                gc_after_trial=True
            )
            joblib.dump(study, f"{study_name}.pkl")
        
        # Now safely access best_value
        if len(study.trials) > 0:
            logging.info(f"Best validation loss: {study.best_value}")
        else:
            logging.error("Hyperparameter optimization failed to produce any trials.")
            return
    
    # Now train the final model with the best parameters
    logging.info("Training final model with best parameters...")
    best_params = study.best_params
    
    # Create model with best parameters
    latent_dim = best_params.get('latent_dim', 96)
    use_attention = best_params.get('use_attention', True)
    dropout_rate = best_params.get('dropout_rate', 0.5)
    residual_layers = best_params.get('residual_layers', 2)
    
    # Reconstruct filters from best params
    filters = []
    num_filters = best_params.get('num_filters', 3)
    for i in range(num_filters):
        if f'filter_{i}' in best_params:
            filters.append(best_params[f'filter_{i}'])
        else:
            # Default fallback values if the parameter isn't found
            filters.append(32 * (2 ** i))
    
    # Create final model
    model = Autoencoder(
        latent_dim=latent_dim,
        use_attention=use_attention,
        dropout_rate=dropout_rate,
        residual_layers=residual_layers,
        filters=tuple(filters)
    ).to(device)
    
    # Print model info
    with torch.no_grad():
        _, encoded = model(torch.zeros(1, 3, 152, 224).to(device))
        input_size = 3 * 152 * 224
        encoded_size = encoded.numel()
        compression_ratio = input_size / encoded_size
        effective_compression = compression_ratio * 2
        logging.info(f"Compression Ratio: {compression_ratio:.2f}")
        logging.info(f"Effective Compression (with dropout): {effective_compression:.2f}")
        logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Set up loss function with optimized parameters
    criterion = HybridLoss(
        device, 
        alpha=best_params.get('alpha', 0.6), 
        beta=best_params.get('beta', 0.35), 
        gamma=best_params.get('gamma', 0.05)
    )
    
    # Set up optimizer based on best parameters
    lr = best_params.get('lr', 0.002)
    weight_decay = best_params.get('weight_decay', 0.0025)
    optimizer_name = best_params.get('optimizer', 'Lion')
    
    if optimizer_name == 'Lion':
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up data loaders with best batch size
    batch_size = best_params.get('batch_size', 128)
    num_workers = min(4, os.cpu_count() or 2)
    prefetch_factor = 2
    accumulation_steps = 2
    
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
    
    # Set up scheduler with best learning rate
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
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
        log_memory_usage()

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
                'hyperparams': best_params,  # Save the hyperparameters too
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
    
    # Save the hyperparameter tuning visualization after training
    try:
        visualize_optuna_results(study_name)
        logging.info("Hyperparameter visualizations saved")
    except Exception as e:
        logging.error(f"Failed to create hyperparameter visualizations: {e}")

if __name__ == '__main__':
    args = parse_args()  
    try:
        train(retrain=args.retrain)  # Pass retrain argument to train function
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()