import os
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms

def load_data(cache_tensors=True, cache_file="processed_data_tensors.pt"):
    """
    Load and preprocess image data from numpy arrays.
    
    Args:
        cache_tensors (bool): Whether to cache processed tensors
        cache_file (str): Path to cache file
    
    Returns:
        torch.utils.data.Dataset: Processed dataset
    """
    # Check for cached data
    if cache_tensors and os.path.exists(cache_file):
        data_tensor = torch.load(cache_file)
        return TensorDataset(data_tensor)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((144, 224)),
    ])

    subsets = ["subset_1.npy", "subset_2.npy", "subset_3.npy"]
    all_tensors = []
    batch_size = 500

    for subset_file in subsets:
        data = np.load(subset_file, mmap_mode='r')
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].reshape(-1, 150, 225, 3)
            tensors = torch.stack([transform(img) for img in batch])
            all_tensors.append(tensors)
            logging.info(f"Processed {len(all_tensors) * batch_size} images")

    # Combine and cache tensors
    data_tensor = torch.cat(all_tensors)
    if cache_tensors:
        torch.save(data_tensor, cache_file)

    return TensorDataset(data_tensor)

def create_dataloaders(dataset, train_ratio=0.9, batch_size=32, num_workers=2):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset (torch.utils.data.Dataset): Input dataset
        train_ratio (float): Proportion of data for training
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader