import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class HybridLoss(nn.Module):
    def __init__(self, device, alpha=0.6, beta=0.35, gamma=0.05):
        super(HybridLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target, encoded=None):
        # Clamp inputs to avoid extreme values
        pred = torch.clamp(pred, 0.001, 0.999)
        
        # Compute losses
        ssim_loss = 1 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)
        
        # Add latent loss if encoded representation is provided
        if encoded is not None:
            latent_loss = torch.mean(torch.abs(encoded) + 1e-8)
            return self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * latent_loss
        else:
            return self.alpha * mse_loss + self.beta * ssim_loss