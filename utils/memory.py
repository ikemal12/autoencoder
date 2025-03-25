import os
import psutil
import torch
import logging

def log_memory_usage():
    """
    Log the current memory usage of the process.
    
    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
    logging.info(f"Memory usage: {memory_usage:.2f} MB")
    return memory_usage

def free_memory():
    """Free up GPU memory and run garbage collection"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logging.warning(f"Error freeing GPU memory: {e}")
    
    try:
        import gc
        gc.collect()
    except Exception as e:
        logging.warning(f"Error running garbage collection: {e}")

def get_optimal_num_workers():
    """
    Determine the optimal number of workers for DataLoader.
    
    Returns:
        int: Recommended number of workers
    """
    return min(4, os.cpu_count() or 2)