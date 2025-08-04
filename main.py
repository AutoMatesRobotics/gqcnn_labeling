import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from utils.preprocessing import prepare_patches
from utils.labeling import labeling_session, extract_patches_by_centers
from utils.exporting import create_labeled_patches_dataset
from utils.visualising import visualise_labeling
from utils.logger import create_logger
from autolab_core import CameraIntrinsics

def main():
    # Parameters
    dataset_name = "example_dataset"
    input_folder = "example_captures"
    root_dir = Path(f"data/input/{input_folder}")
    save_dir = Path(f"data/output/{dataset_name}")
    patch_size = 96
    stride = 10

    # Create logger for this session
    log_dir = save_dir / "log"
    logger = create_logger(dataset_name, log_dir=str(log_dir))
        
    try:
        # Log session parameters
        logger.log_info(f"Session parameters:")
        logger.log_info(f"  Root directory: {root_dir}")
        logger.log_info(f"  Save directory: {save_dir}")
        logger.log_info(f"  Patch size: {patch_size}")
        logger.log_info(f"  Stride: {stride}")
        
        # Count total folders to process
        folders = [f for f in root_dir.iterdir() if f.is_dir()]
        logger.log_info(f"Found {len(folders)} folders to process")
        
        session_start_time = time.time()
        successful_captures = 0
        failed_captures = 0
        total_good_patches = 0
        total_bad_patches = 0
        
        # Loop through each folder inside root_dir
        index = 0
        for folder in folders:
            index += 1
            capture_start_time = time.time()
            
            if folder.is_dir():
                depth_file_path = folder / "depth_0.npy"
                camera_intr_path = folder / "zivid.intr"
                
                # Check if required files exist
                if depth_file_path.exists() and camera_intr_path.exists():
                    logger.log_capture_start(index, folder.name)
                    logger.log_capture_files(depth_file_path, camera_intr_path)

                    try:
                        # Stage 1: Preparing Patches From Scene
                        logger.log_labeling_stage_start("Patch Preparation")
                        
                        patches, patch_centers, raw_depth_clean = prepare_patches(
                            depth_file_path, camera_intr_path, 
                            patch_size=patch_size, stride=stride
                        )
                        
                        # Log patch preparation results
                        logger.log_patch_preparation(
                            patch_size, stride, len(patches), raw_depth_clean.shape
                        )
                        
                        # Log depth preprocessing info (you may need to modify prepare_patches to return this info)
                        depth_data = np.load(depth_file_path)
                        original_range = (depth_data.min(), depth_data.max())
                        cleaned_range = (raw_depth_clean.min(), raw_depth_clean.max())
                        logger.log_depth_preprocessing(original_range, cleaned_range, 
                                                     holes_filled=True, outliers_clipped=True)
                        
                        # Stage 2: Labeling Patches Manually
                        logger.log_labeling_stage_start("Manual Labeling")
                        
                        labeling_start_time = time.time()
                        good_patch_centers, bad_patch_centers = labeling_session(
                            raw_depth_clean, patch_centers, logger=logger
                        )
                        labeling_end_time = time.time()
                        
                        # Log labeling results
                        total_patches_in_boxes = len(good_patch_centers) + len(bad_patch_centers)
                        logger.log_patches_in_boxes(len(patch_centers), total_patches_in_boxes)
                        logger.log_grasp_labeling_results(
                            len(good_patch_centers), len(bad_patch_centers), len(patch_centers)
                        )
                        logger.log_performance_metrics(
                            labeling_end_time - labeling_start_time,
                            patches_per_second=total_patches_in_boxes / (labeling_end_time - labeling_start_time) if (labeling_end_time - labeling_start_time) > 0 else None
                        )
                        
                        # Log user interaction summary
                        logger.log_user_interaction("Labeling Complete", 
                                                   f"Selected {len(good_patch_centers)} good and {len(bad_patch_centers)} bad patches")
                        
                        # Stage 3: Visualize Labeling
                        logger.log_info("Generating labeling visualization...")
                        vis_save_path = log_dir / f"labeled_scene_{folder.name}.png"
                        visualise_labeling(raw_depth_clean, good_patch_centers, bad_patch_centers, save_path=str(vis_save_path), logger=logger
                        )
                        
                        # Stage 4: Creating Patches From Labeled Data
                        logger.log_labeling_stage_start("Patch Extraction")
                        
                        good_patches, bad_patches, good_indices, bad_indices = extract_patches_by_centers(
                            patches, patch_centers, good_patch_centers, bad_patch_centers, logger=logger
                        )
                        
                        # Stage 5: Creating Labeled Patches Dataset
                        logger.log_labeling_stage_start("Dataset Creation")
                        
                        tensor_data = create_labeled_patches_dataset(
                            good_patches=good_patches,
                            bad_patches=bad_patches,
                            good_patch_centers=good_patch_centers,
                            bad_patch_centers=bad_patch_centers,
                            patch_size=patch_size,
                            camera_intr=CameraIntrinsics.load(camera_intr_path),
                            save_dir=str(save_dir),
                            tensor_idx=index,
                            dataset_name=dataset_name
                        )
                        
                        if tensor_data:
                            logger.log_dataset_creation(
                                len(good_patches), len(bad_patches),
                                tensor_data['tf_depth_ims'].shape, save_dir
                            )
                        
                        # Calculate processing time for this capture
                        capture_end_time = time.time()
                        capture_duration = capture_end_time - capture_start_time
                        
                        # Log successful completion
                        logger.log_capture_success(index, len(good_patches), len(bad_patches))
                        logger.log_performance_metrics(capture_duration)
                        
                        # Update totals
                        successful_captures += 1
                        total_good_patches += len(good_patches)
                        total_bad_patches += len(bad_patches)
                    
                    except Exception as e:
                        logger.log_capture_error(index, folder.name, e)
                        failed_captures += 1
                        continue
                        
                else:
                    missing_files = []
                    if not depth_file_path.exists():
                        missing_files.append("depth_0.npy")
                    if not camera_intr_path.exists():
                        missing_files.append("zivid.intr")
                    
                    logger.log_warning(f"Skipping {folder.name} - missing files: {', '.join(missing_files)}")
                    failed_captures += 1
        
        # Log session summary
        session_end_time = time.time()
        total_session_time = session_end_time - session_start_time
        
        logger.log_info("="*60)
        logger.log_info("SESSION SUMMARY")
        logger.log_info("="*60)
        logger.log_info(f"Total folders processed: {len(folders)}")
        logger.log_info(f"Successful captures: {successful_captures}")
        logger.log_info(f"Failed captures: {failed_captures}")
        logger.log_info(f"Success rate: {(successful_captures/len(folders)*100):.1f}%")
        logger.log_info(f"Total good patches labeled: {total_good_patches}")
        logger.log_info(f"Total bad patches labeled: {total_bad_patches}")
        logger.log_info(f"Total patches labeled: {total_good_patches + total_bad_patches}")
        if total_good_patches + total_bad_patches > 0:
            logger.log_info(f"Overall good patch percentage: {(total_good_patches/(total_good_patches + total_bad_patches)*100):.1f}%")
        logger.log_info(f"Total session time: {total_session_time/60:.1f} minutes")
        if successful_captures > 0:
            logger.log_info(f"Average time per capture: {total_session_time/successful_captures:.1f} seconds")
        
        logger.log_session_end()
        
    except KeyboardInterrupt:
        logger.log_warning("Session interrupted by user (Ctrl+C)")
        logger.log_session_end()
        
    except Exception as e:
        logger.log_error(f"Unexpected error during session: {str(e)}")
        logger.log_session_end()
        raise
        
    finally:
        # Always close the logger
        logger.close()
        print(f"\nLog file saved to: {logger.get_log_path()}")


if __name__ == "__main__":
    main()