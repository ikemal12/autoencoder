import os
import time
import torch
import logging
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from lion_pytorch import Lion
from models.loss import HybridLoss
from utils.visualisation import visualize_results

def train_model(model, train_loader, val_loader, device, best_params):
    """
    Train the autoencoder model.
    
    Args:
        model (torch.nn.Module): Autoencoder model
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device to train on
        best_params (dict): Best hyperparameters from optimization
    """
    lr = best_params.get('lr', 0.002)
    weight_decay = best_params.get('weight_decay', 0.0025)

    criterion = HybridLoss(
        device, 
        alpha=best_params.get('alpha', 0.6), 
        beta=best_params.get('beta', 0.35), 
        gamma=best_params.get('gamma', 0.05)
    )

    optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    num_epochs = 20
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            img = batch[0].to(device)
            
            # Forward pass
            recon, encoded = model(img)
            loss = criterion(recon, img, encoded)
            
            # Backward pass and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_ssim = 0
        
        with torch.no_grad():
            for batch in val_loader:
                img = batch[0].to(device)
                recon, encoded = model(img)
                loss = criterion(recon, img, encoded)
                ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(recon, img)
                
                val_loss += loss.item()
                val_ssim += ssim.item()
        
        # Logging and checkpointing
        val_loss /= len(val_loader)
        val_ssim /= len(val_loader)
        
        logging.info(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}, SSIM: {val_ssim:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        visualize_results(model, val_loader, device, epoch)

    logging.info("Training completed.")