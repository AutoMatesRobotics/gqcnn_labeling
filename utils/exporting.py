import numpy as np
import os
from pathlib import Path

def create_labeled_patches_dataset(good_patches, bad_patches, 
                                 good_patch_centers, bad_patch_centers,
                                 patch_size=96, 
                                 camera_intr=None,
                                 save_dir="data/labeled_patches_tensors",
                                 tensor_idx=0,
                                 dataset_name="manual_labels",
                                 logger=None):
    """
    Create GQ-CNN compatible tensors from manually labeled patches
    
    Parameters:
    -----------
    good_patches : np.ndarray
        Array of good grasp patches, shape (N_good, H, W) or (N_good, H, W, 1)
    bad_patches : np.ndarray  
        Array of bad grasp patches, shape (N_bad, H, W) or (N_bad, H, W, 1)
    good_patch_centers : list
        List of (x, y) coordinates for good patches
    bad_patch_centers : list
        List of (x, y) coordinates for bad patches
    patch_size : int
        Size of patches (should match patch dimensions)
    camera_intr : CameraIntrinsics object
        Camera intrinsics (optional)
    save_dir : str
        Directory to save tensors
    tensor_idx : int
        Tensor file index
    dataset_name : str
        Name/description of the dataset
    logger : DataLabelingLogger, optional
        Logger instance for recording operations
        
    Returns:
    --------
    dict : Dictionary containing the tensor data
    """
    
    if logger:
        logger.log_info(f"Creating labeled patches dataset...")
        logger.log_info(f"Good patches: {len(good_patches)}")
        logger.log_info(f"Bad patches: {len(bad_patches)}")
    else:
        print(f"Creating labeled patches dataset...")
        print(f"Good patches: {len(good_patches)}")
        print(f"Bad patches: {len(bad_patches)}")
    
    # Combine patches and create labels
    all_patches = []
    all_scores = []
    all_centers = []
    
    # Add good patches (label = 1.0)
    if len(good_patches) > 0:
        for i, patch in enumerate(good_patches):
            # Ensure patch has channel dimension
            if patch.ndim == 2:
                patch = patch[:, :, np.newaxis]
            elif patch.ndim == 3 and patch.shape[2] != 1:
                patch = patch[:, :, 0:1]  # Take first channel only
                
            all_patches.append(patch)
            all_scores.append(1.0)  # Good grasp
            all_centers.append(good_patch_centers[i])
    
    # Add bad patches (label = 0.0)
    if len(bad_patches) > 0:
        for i, patch in enumerate(bad_patches):
            # Ensure patch has channel dimension
            if patch.ndim == 2:
                patch = patch[:, :, np.newaxis]
            elif patch.ndim == 3 and patch.shape[2] != 1:
                patch = patch[:, :, 0:1]  # Take first channel only
                
            all_patches.append(patch)
            all_scores.append(0.0)  # Bad grasp
            all_centers.append(bad_patch_centers[i])
    
    # Convert to numpy arrays (GQ-CNN tensor format)
    if len(all_patches) > 0:
        tf_depth_ims = np.array(all_patches, dtype=np.float32)  # Shape: (N, H, W, 1)
        grasp_metrics = np.array(all_scores, dtype=np.float32)   # Shape: (N,)
        
        # Log tensor creation details
        if logger:
            logger.log_info(f"Labeled patches tensor creation complete:")
            logger.log_info(f"  tf_depth_ims shape: {tf_depth_ims.shape}")
            logger.log_info(f"  grasp_metrics shape: {grasp_metrics.shape}")
            logger.log_info(f"  Depth range: {tf_depth_ims.min():.3f} to {tf_depth_ims.max():.3f}")
            logger.log_info(f"  Quality range: {grasp_metrics.min():.3f} to {grasp_metrics.max():.3f}")
        else:
            print(f"\nLabeled patches tensor creation complete:")
            print(f"  tf_depth_ims shape: {tf_depth_ims.shape}")
            print(f"  grasp_metrics shape: {grasp_metrics.shape}")
            print(f"  Depth range: {tf_depth_ims.min():.3f} to {tf_depth_ims.max():.3f}")
            print(f"  Quality range: {grasp_metrics.min():.3f} to {grasp_metrics.max():.3f}")
        
        # Analyze label distribution
        num_good = np.sum(grasp_metrics > 0.5)
        num_bad = np.sum(grasp_metrics <= 0.5)
        good_percentage = num_good/(num_good+num_bad)*100 if (num_good+num_bad) > 0 else 0
        
        if logger:
            logger.log_info(f"  Label distribution: {num_good} good, {num_bad} bad ({good_percentage:.1f}% good)")
        else:
            print(f"  Label distribution: {num_good} good, {num_bad} bad ({good_percentage:.1f}% good)")
        
        # Get camera intrinsics if available
        if camera_intr is not None:
            camera_intrinsics = np.array([camera_intr.fx, camera_intr.fy, camera_intr.cx, camera_intr.cy])
            if logger:
                logger.log_info(f"  Camera intrinsics: fx={camera_intr.fx:.1f}, fy={camera_intr.fy:.1f}, cx={camera_intr.cx:.1f}, cy={camera_intr.cy:.1f}")
        else:
            # Default values if not provided
            camera_intrinsics = np.array([patch_size, patch_size, patch_size/2, patch_size/2])
            if logger:
                logger.log_warning("No camera intrinsics provided, using default values")
        
        tensor_data = {
            'tf_depth_ims': tf_depth_ims,
            'grasp_metrics': grasp_metrics,
            'patch_centers': all_centers,
            'patch_size': patch_size,
            'camera_intrinsics': camera_intrinsics,
            'method': 'manual_labeling',
            'dataset_name': dataset_name,
            'num_good': int(num_good),
            'num_bad': int(num_bad)
        }
        
        # Save the tensors
        save_labeled_tensors(tensor_data, save_dir, tensor_idx, logger)
        
        return tensor_data
    
    else:
        error_msg = "No patches to save!"
        if logger:
            logger.log_error(error_msg)
        else:
            print(error_msg)
        return None


def save_labeled_tensors(tensor_data, save_dir="data/labeled_patches_tensors", tensor_idx=0, logger=None):
    """
    Save labeled GQ-CNN tensors in the standard format
    
    Parameters:
    -----------
    tensor_data : dict
        Dictionary containing tensors and metadata
    save_dir : str
        Directory to save tensors
    tensor_idx : int
        Tensor file index
    logger : DataLabelingLogger, optional
        Logger instance for recording operations
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main tensor files (same format as training expects)
    tf_depth_path = os.path.join(save_dir, f"tf_depth_ims_{tensor_idx}.npz")
    metrics_path = os.path.join(save_dir, f"grasp_metrics_{tensor_idx}.npz")
    
    np.savez_compressed(tf_depth_path, tensor_data['tf_depth_ims'])
    np.savez_compressed(metrics_path, tensor_data['grasp_metrics'])
    
    # Save metadata with additional info
    metadata_path = os.path.join(save_dir, f"metadata_{tensor_idx}.npz")
    np.savez_compressed(
        metadata_path,
        patch_size=tensor_data['patch_size'],
        camera_intrinsics=tensor_data['camera_intrinsics'],
        num_samples=len(tensor_data['tf_depth_ims']),
        method=tensor_data['method'],
        dataset_name=tensor_data['dataset_name'],
        num_good=tensor_data['num_good'],
        num_bad=tensor_data['num_bad'],
        patch_centers=tensor_data['patch_centers']
    )
    
    # Calculate file sizes
    tf_size = os.path.getsize(tf_depth_path) / 1024 / 1024
    metrics_size = os.path.getsize(metrics_path) / 1024 / 1024
    metadata_size = os.path.getsize(metadata_path) / 1024 / 1024
    total_size = tf_size + metrics_size + metadata_size
    
    if logger:
        logger.log_info(f"Tensors saved to {save_dir}:")
        logger.log_file_save(tf_depth_path, tf_size)
        logger.log_file_save(metrics_path, metrics_size)
        logger.log_file_save(metadata_path, metadata_size)
        logger.log_info(f"Total file size: {total_size:.2f} MB")
    else:
        print(f"\nTensors saved to {save_dir}:")
        print(f"  {tf_depth_path}")
        print(f"  {metrics_path}")
        print(f"  {metadata_path}")
        print(f"  File sizes:")
        print(f"    Images: {tf_size:.2f} MB")
        print(f"    Metrics: {metrics_size:.2f} MB") 
        print(f"    Metadata: {metadata_size:.2f} MB")
        print(f"    Total: {total_size:.2f} MB")


def load_labeled_tensors(tensors_dir, tensor_idx=0, logger=None):
    """
    Load and test the created labeled tensors
    
    Parameters:
    -----------
    tensors_dir : str
        Directory containing tensors
    tensor_idx : int
        Tensor file index to load
    logger : DataLabelingLogger, optional
        Logger instance for recording operations
        
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
            if logger:
                logger.log_info(f"Metadata loaded:")
                for key in metadata.files:
                    value = metadata[key]
                    if isinstance(value, np.ndarray) and value.size == 1:
                        logger.log_info(f"  {key}: {value.item()}")
                    elif isinstance(value, np.ndarray) and value.size <= 4:
                        logger.log_info(f"  {key}: {value}")
                    else:
                        logger.log_info(f"  {key}: {type(value)} (size: {value.shape if hasattr(value, 'shape') else len(value)})")
            else:
                print(f"Metadata loaded:")
                for key in metadata.files:
                    value = metadata[key]
                    if isinstance(value, np.ndarray) and value.size == 1:
                        print(f"  {key}: {value.item()}")
                    elif isinstance(value, np.ndarray) and value.size <= 4:
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {type(value)} (size: {value.shape if hasattr(value, 'shape') else len(value)})")
        
        num_good = np.sum(metrics > 0.5)
        num_bad = np.sum(metrics <= 0.5)
        
        if logger:
            logger.log_info(f"Labeled data loaded successfully!")
            logger.log_info(f"Images shape: {images.shape}")
            logger.log_info(f"Metrics shape: {metrics.shape}")
            logger.log_info(f"Label distribution: {num_good} good, {num_bad} bad")
            logger.log_info(f"Depth range: {images.min():.3f} to {images.max():.3f}")
        else:
            print(f"\nLabeled data loaded successfully!")
            print(f"Images shape: {images.shape}")
            print(f"Metrics shape: {metrics.shape}")
            print(f"Label distribution: {num_good} good, {num_bad} bad")
            print(f"Depth range: {images.min():.3f} to {images.max():.3f}")
        
        return images, metrics, metadata
        
    except Exception as e:
        error_msg = f"Error loading labeled data: {e}"
        if logger:
            logger.log_error(error_msg)
            logger.log_info(f"Make sure you have these files in {tensors_dir}/:")
            logger.log_info(f"- tf_depth_ims_{tensor_idx}.npz")
            logger.log_info(f"- grasp_metrics_{tensor_idx}.npz")
            logger.log_info(f"- metadata_{tensor_idx}.npz (optional)")
        else:
            print(error_msg)
            print(f"Make sure you have these files in {tensors_dir}/:")
            print(f"- tf_depth_ims_{tensor_idx}.npz")
            print(f"- grasp_metrics_{tensor_idx}.npz")
            print(f"- metadata_{tensor_idx}.npz (optional)")
        return None, None, None
