"""
Device management utilities.
"""
import random
import numpy as np
import torch


def get_device(prefer_gpu: bool = True, gpu_id: int = None) -> torch.device:
    """
    Get the best available device.
    
    Supports:
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs, uses cuda interface)
    - MPS (Apple Silicon)
    - CPU fallback
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        gpu_id: Specific GPU ID to use (None = auto-select best discrete GPU)
        
    Returns:
        Torch device
    """
    if not prefer_gpu:
        print("Using CPU (GPU disabled)")
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        # Auto-select best GPU if not specified
        if gpu_id is None:
            # Try to find a discrete GPU (skip integrated graphics)
            best_gpu = 0
            best_memory = 0
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_name = torch.cuda.get_device_name(i)
                # Prefer discrete GPUs (more memory, specific names)
                if props.total_memory > best_memory:
                    # Skip integrated graphics (usually has less memory)
                    if "Radeon(TM) Graphics" not in gpu_name or num_gpus == 1:
                        best_gpu = i
                        best_memory = props.total_memory
            gpu_id = best_gpu
        
        device = torch.device(f"cuda:{gpu_id}")
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        
        # Test GPU with a simple operation to verify it works
        try:
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T
            del test_tensor
            print(f"Using GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
            return device
        except Exception as e:
            print(f"GPU {gpu_id} ({gpu_name}) failed test: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
        return device
    
    print("Using CPU (no GPU available)")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['cuda']:
        info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['cuda_devices'].append({
                'name': props.name,
                'total_memory_gb': props.total_memory / 1e9,
                'compute_capability': f"{props.major}.{props.minor}",
            })
    
    return info


def memory_stats(device: torch.device) -> dict:
    """
    Get memory statistics for a device.
    
    Args:
        device: Torch device
        
    Returns:
        Dictionary with memory statistics
    """
    if device.type == 'cuda':
        return {
            'allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
        }
    return {}


def clear_memory(device: torch.device) -> None:
    """
    Clear GPU memory cache.
    
    Args:
        device: Torch device
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    print("Device Information:")
    info = get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nSelecting device...")
    device = get_device()
    
    if device.type == 'cuda':
        print("\nMemory stats:")
        stats = memory_stats(device)
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")
