import numpy as np
import matplotlib.pyplot as plt
from utils.loading import load_labeled_tensor

import os
import matplotlib.pyplot as plt

def visualise_labeling(depth_image, good_patch_centers, bad_patch_centers, save_path=None, logger=None):
    """ 
    Visualize labeled patches on the original depth image

    Args:
        depth_image (np.ndarray): The original depth image.
        good_patch_centers (list): List of (x, y) coordinates for good patches.
        bad_patch_centers (list): List of (x, y) coordinates for bad patches.
        save_dir (str, optional): Directory path to save the plot. If None, the plot is just shown.
    """

    # Separate patch centers by prediction
    good_centers_x, good_centers_y = zip(*good_patch_centers) if good_patch_centers else ([], [])
    bad_centers_x, bad_centers_y = zip(*bad_patch_centers) if bad_patch_centers else ([], [])

    plt.figure(figsize=(12, 8))
    plt.imshow(depth_image)

    if bad_centers_x:
        plt.scatter(bad_centers_x, bad_centers_y, c='red', s=15, alpha=0.7, label=f'Bad Grasp ({len(bad_centers_x)})')
    if good_centers_x:
        plt.scatter(good_centers_x, good_centers_y, c='green', s=15, alpha=0.7, label=f'Good Grasp ({len(good_centers_x)})')

    plt.title(f'Depth Image with Grasp Quality Predictions\n'
              f'Green: Good Grasps ({len(good_centers_x)}), Red: Bad Grasps ({len(bad_centers_x)})')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.colorbar(label='Depth (normalized)')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        if logger:
            logger.log_info(f"Plot saved to {save_path}")
        else:
            print(f"Plot saved to {save_path}")
        plt.show()
        plt.close()
    else:
        plt.show()



def visualize_saved_tensor(tensors_dir, tensor_idx=0, num_samples=8, logger=None):
    """
    Visualize samples from saved tensors
    """
    import matplotlib.pyplot as plt
    
    if logger:
        logger.log_info(f"Visualizing saved tensors from {tensors_dir}")
    
    images, metrics, metadata = load_labeled_tensor(tensors_dir, tensor_idx, logger)
    
    if images is None:
        return
    
    # Separate good and bad samples
    good_mask = metrics > 0.5
    bad_mask = metrics <= 0.5
    
    good_images = images[good_mask]
    bad_images = images[bad_mask]
    
    if logger:
        logger.log_info(f"Creating visualization with {num_samples} sample patches")
    
    # Plot samples
    fig, axes = plt.subplots(2, num_samples//2, figsize=(12, 6))
    
    # Plot good samples
    if len(good_images) > 0:
        good_indices = np.random.choice(len(good_images), min(num_samples//2, len(good_images)), replace=False)
        for i, idx in enumerate(good_indices):
            axes[0, i].imshow(good_images[idx, :, :, 0], cmap='gray')
            axes[0, i].set_title(f'Good {idx}')
            axes[0, i].axis('off')
            # Add green center dot
            center = good_images[idx].shape[0] // 2
            axes[0, i].plot(center, center, 'g.', markersize=8)
    
    # Plot bad samples
    if len(bad_images) > 0:
        bad_indices = np.random.choice(len(bad_images), min(num_samples//2, len(bad_images)), replace=False)
        for i, idx in enumerate(bad_indices):
            axes[1, i].imshow(bad_images[idx, :, :, 0], cmap='gray')
            axes[1, i].set_title(f'Bad {idx}')
            axes[1, i].axis('off')
            # Add red center dot  
            center = bad_images[idx].shape[0] // 2
            axes[1, i].plot(center, center, 'r.', markersize=8)
    
    plt.suptitle(f'Saved Labeled Patches (Tensor {tensor_idx})')
    plt.tight_layout()
    plt.show()
    
    if logger:
        logger.log_info("Visualization completed")