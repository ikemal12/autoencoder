import torch
import matplotlib.pyplot as plt
import numpy as np
import optuna.visualization as vis
import joblib
import logging

def visualize_results(model, val_loader, device, epoch, num_images=5):
    """
    Visualize model results for a batch of images.
    
    Args:
        model (torch.nn.Module): Trained autoencoder model
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device to run inference on
        epoch (int): Current training epoch
        num_images (int, optional): Number of images to visualize. Defaults to 5.
    """
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
        plt.savefig(f'results_epoch_{epoch+1}.png')
        plt.close()

def visualize_optuna_results(study_name="autoencoder_hparam_study"):
    """
    Visualize hyperparameter optimization results from Optuna.
    
    Args:
        study_name (str, optional): Name of the Optuna study. Defaults to "autoencoder_hparam_study".
    """
    try:
        # Load the study
        study = joblib.load(f"{study_name}.pkl")
        
        # Create a directory for visualizations if it doesn't exist
        import os
        os.makedirs("optuna_visualizations", exist_ok=True)
        
        # Plot optimization history
        fig1 = vis.plot_optimization_history(study)
        fig1.write_image("optuna_visualizations/optimization_history.png")
        
        # Plot parameter importances
        fig2 = vis.plot_param_importances(study)
        fig2.write_image("optuna_visualizations/param_importances.png")
        
        # Plot parallel coordinates
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_image("optuna_visualizations/parallel_coordinate.png")
        
        # Plot slice plots for specific parameters
        for param in ['latent_dim', 'dropout_rate', 'lr']:
            if param in study.best_params:
                fig = vis.plot_slice(study, params=[param])
                fig.write_image(f"optuna_visualizations/slice_{param}.png")
        
        logging.info("Optuna visualization plots saved in 'optuna_visualizations' directory")
    
    except Exception as e:
        logging.error(f"Failed to create hyperparameter visualizations: {e}")