import torch
import logging
import optuna
from models.autoencoder import Autoencoder
from models.loss import HybridLoss
from data.dataset import load_data, create_dataloaders
from training.trainer import train_model
from training.tuning import optimize_hyperparameters

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    logging.info("Loading dataset...")
    dataset = load_data()
    train_loader, val_loader = create_dataloaders(dataset)

    # Hyperparameter optimization (optional)
    best_params = optimize_hyperparameters(
        train_loader, 
        val_loader, 
        device, 
        num_trials=10
    )

    # Create model with best parameters
    model = Autoencoder(
        latent_dim=best_params.get('latent_dim', 96),
        use_attention=best_params.get('use_attention', True),
        dropout_rate=best_params.get('dropout_rate', 0.5),
        residual_layers=best_params.get('residual_layers', 2),
        filters=tuple(best_params.get('filters', [32, 64, 128]))
    ).to(device)

    # Train the model
    train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        best_params
    )

if __name__ == '__main__':
    main()