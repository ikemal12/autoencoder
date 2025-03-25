import optuna
import torch
from models.autoencoder import Autoencoder
from models.loss import HybridLoss

def objective(trial, train_loader, val_loader, device):
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): Optimization trial
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device to train on
    
    Returns:
        float: Validation loss
    """
    # Hyperparameters to optimise
    latent_dim = trial.suggest_int('latent_dim', 48, 196, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    residual_layers = trial.suggest_int('residual_layers', 1, 3)
    
    filters = []
    num_filters = trial.suggest_int('num_filters', 2, 4)
    for i in range(num_filters):
        filters.append(trial.suggest_int(f'filter_{i}', 16, 256, step=16))
    
    model = Autoencoder(
        latent_dim=latent_dim, 
        use_attention=use_attention,
        dropout_rate=dropout_rate,
        residual_layers=residual_layers,
        filters=tuple(filters)
    ).to(device)
    
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = HybridLoss(
        device, 
        alpha=trial.suggest_float('alpha', 0.3, 0.8),
        beta=trial.suggest_float('beta', 0.1, 0.6),
        gamma=trial.suggest_float('gamma', 0.01, 0.1)
    )
    
    # Training loop
    model.train()
    val_loss = 0
    
    for batch in val_loader:
        img = batch[0].to(device)
        optimizer.zero_grad()
        
        recon, encoded = model(img)
        loss = criterion(recon, img, encoded)
        
        loss.backward()
        optimizer.step()
        
        val_loss += loss.item()
    
    return val_loss / len(val_loader)

def optimize_hyperparameters(train_loader, val_loader, device, num_trials=10):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device to train on
        num_trials (int): Number of optimization trials
    
    Returns:
        dict: Best hyperparameters
    """
    study = optuna.create_study(direction='minimize')
    objective_with_args = lambda trial: objective(
        trial, train_loader, val_loader, device
    )
    
    study.optimize(objective_with_args, n_trials=num_trials)
    
    return study.best_params