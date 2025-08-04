import numpy as np
from autolab_core import (CameraIntrinsics, DepthImage)


def prepare_patches(depth_file_path, camera_intr_path, normalise=True, patch_size=96, stride=10, transfrom=True):
    """
    Prepares patches from a depth image for GQCNN labeling.
    Args:
        depth_file_path (str): Path to the depth image file.
        camera_intr_path (str): Path to the camera intrinsics file.
        normalise (bool): Whether to normalise the depth values.
        patch_size (int): Size of each square patch.
        stride (int): Stride for extracting patches.
    """
    print(f"Preparing patches from depth file: {depth_file_path}")
    
    # Load camera intrinsics
    camera_intr = CameraIntrinsics.load(camera_intr_path)
    # Load depth image
    depth_data = np.load(depth_file_path)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    print(f"Original depth image shape: {depth_im.shape}")
    print(f"Depth range: {depth_data.min():.3f} to {depth_data.max():.3f}")
    
    raw_depth = depth_im._data
    
    # Inpaint holes and remove outliers
    raw_depth_clean = raw_depth.copy()
    
    # Fill holes (zeros) with median filter
    if np.any(raw_depth_clean == 0):
        from scipy import ndimage
        print("Filling holes with median filter...")
        mask = raw_depth_clean == 0
        raw_depth_clean[mask] = ndimage.median_filter(raw_depth_clean, size=5)[mask]
    
    # Remove outliers - use percentile clipping
    valid_pixels = raw_depth_clean[raw_depth_clean > 0]
    if len(valid_pixels) > 0:
        # Use 1st and 99th percentiles to remove extreme outliers
        p1 = np.percentile(valid_pixels, 1)
        p99 = np.percentile(valid_pixels, 99)
        print(f"Clipping outliers: keeping values between {p1:.3f} and {p99:.3f}")
        
        # Clip extreme values
        raw_depth_clean = np.clip(raw_depth_clean, p1, p99)

    if normalise:
        # Normalise
        if raw_depth_clean.max() > raw_depth_clean.min():
            raw_depth_clean = (raw_depth_clean - raw_depth_clean.min()) / (raw_depth_clean.max() - raw_depth_clean.min())
    
    print(f"Cleaned depth value range: {raw_depth_clean.min():.3f} to {raw_depth_clean.max():.3f}")
    
    # Get image dimensions
    height, width = raw_depth_clean.shape[:2]
    
    # Calculate how many patches we can extract
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1
    
    print(f"Image size: {width}x{height}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride}")
    print(f"Number of patches: {num_patches_w} x {num_patches_h} = {num_patches_w * num_patches_h}")
    
    # Extract all patches with depth frame transformation
    patches = []
    patch_centers = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch position
            y_start = i * stride
            x_start = j * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # Extract patch
            patch = raw_depth_clean[y_start:y_end, x_start:x_end]
            
            # Get grasp depth (depth at center of patch)
            center_x = x_start + patch_size // 2
            center_y = y_start + patch_size // 2
            grasp_depth = raw_depth_clean[center_y, center_x]
            
            if transfrom:
                # Transform depth to grasp frame of reference
                patch_transformed = patch - grasp_depth
                
                # Store transformed patch and its center coordinates
                patches.append(patch_transformed)
            else:
                # Store transformed patch and its center coordinates
                patches.append(patch)
                
            patch_centers.append((center_x, center_y))
    
    return patches, patch_centers, raw_depth_clean