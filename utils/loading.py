import numpy as np
import os
from pathlib import Path


def load_labeled_tensor(tensors_dir, tensor_idx=0):
    """
    Load and test the created labeled tensors
    
    Parameters:
    -----------
    tensors_dir : str
        Directory containing tensors
    tensor_idx : int
        Tensor file index to load
        
    Returns:
    --------
    tuple : (images, metrics, metadata) or (None, None, None) if failed
    """
    
    tensors_dir = Path(tensors_dir)
    
    try:
        # Load the main files
        images = np.load(tensors_dir / f"tf_depth_ims_{tensor_idx}.npz")['arr_0']
        metrics = np.load(tensors_dir / f"grasp_metrics_{tensor_idx}.npz")['arr_0']
        
        # Load metadata if available
        metadata_path = tensors_dir / f"metadata_{tensor_idx}.npz"
        metadata = None
        if metadata_path.exists():
            metadata = np.load(metadata_path)
            print(f"Metadata loaded:")
            for key in metadata.files:
                value = metadata[key]
                if isinstance(value, np.ndarray) and value.size == 1:
                    print(f"  {key}: {value.item()}")
                elif isinstance(value, np.ndarray) and value.size <= 4:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value)} (size: {value.shape if hasattr(value, 'shape') else len(value)})")
        
        print(f"\nLabeled data loaded successfully!")
        print(f"Images shape: {images.shape}")
        print(f"Metrics shape: {metrics.shape}")
        print(f"Label distribution: {np.sum(metrics > 0.5)} good, {np.sum(metrics <= 0.5)} bad")
        print(f"Depth range: {images.min():.3f} to {images.max():.3f}")
        
        return images, metrics, metadata
        
    except Exception as e:
        print(f"Error loading labeled data: {e}")
        print(f"Make sure you have these files in {tensors_dir}/:")
        print(f"- tf_depth_ims_{tensor_idx}.npz")
        print(f"- grasp_metrics_{tensor_idx}.npz")
        print(f"- metadata_{tensor_idx}.npz (optional)")
        return None, None, None
