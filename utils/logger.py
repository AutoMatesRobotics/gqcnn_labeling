import logging
import os
from datetime import datetime
from pathlib import Path
import sys


class DataLabelingLogger:
    """
    Comprehensive logger for data labeling sessions.
    Creates timestamped log files and provides structured logging for the entire labeling pipeline.
    """
    
    def __init__(self, dataset_name, log_dir="logs", log_level=logging.INFO):
        """
        Initialize the data labeling logger.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset being labeled
        log_dir : str
            Directory to save log files
        log_level : int
            Logging level (INFO, DEBUG, WARNING, ERROR)
        """
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log filename
        self.log_filename = f"log_{dataset_name}_{self.timestamp}.txt"
        self.log_path = self.log_dir / self.log_filename
        
        # Setup logger
        self.logger = logging.getLogger(f"DataLabeling_{dataset_name}_{self.timestamp}")
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler (also log to console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log session start
        self.log_session_start()
    
    def log_session_start(self):
        """Log the start of a labeling session."""
        self.logger.info("="*80)
        self.logger.info(f"DATA LABELING SESSION STARTED")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Log file: {self.log_path}")
        self.logger.info("="*80)
    
    def log_session_end(self):
        """Log the end of a labeling session."""
        self.logger.info("="*80)
        self.logger.info("DATA LABELING SESSION COMPLETED")
        self.logger.info(f"Total session duration: {self._get_session_duration()}")
        self.logger.info("="*80)
    
    def log_capture_start(self, capture_index, folder_name):
        """Log the start of processing a capture."""
        self.logger.info("-"*60)
        self.logger.info(f"PROCESSING CAPTURE {capture_index}: {folder_name}")
        self.logger.info("-"*60)
    
    def log_capture_files(self, depth_file, intrinsics_file):
        """Log the files being processed for a capture."""
        self.logger.info(f"Depth file: {depth_file}")
        self.logger.info(f"Intrinsics file: {intrinsics_file}")
    
    def log_patch_preparation(self, patch_size, stride, num_patches, image_shape):
        """Log patch preparation parameters and results."""
        self.logger.info("--- Patch Preparation ---")
        self.logger.info(f"Image shape: {image_shape}")
        self.logger.info(f"Patch size: {patch_size}x{patch_size}")
        self.logger.info(f"Stride: {stride}")
        self.logger.info(f"Total patches extracted: {num_patches}")
    
    def log_depth_preprocessing(self, original_range, cleaned_range, holes_filled=False, outliers_clipped=False):
        """Log depth preprocessing results."""
        self.logger.info("--- Depth Preprocessing ---")
        self.logger.info(f"Original depth range: {original_range[0]:.3f} to {original_range[1]:.3f}")
        self.logger.info(f"Cleaned depth range: {cleaned_range[0]:.3f} to {cleaned_range[1]:.3f}")
        if holes_filled:
            self.logger.info("Holes filled with median filter")
        if outliers_clipped:
            self.logger.info("Outliers clipped using percentile method")
    
    def log_labeling_stage_start(self, stage_name):
        """Log the start of a labeling stage."""
        self.logger.info(f"--- {stage_name.upper()} STAGE STARTED ---")
    
    def log_object_detection(self, num_boxes):
        """Log object detection results."""
        self.logger.info(f"Object boxes drawn: {num_boxes}")
        if num_boxes == 0:
            self.logger.warning("No object boxes were drawn - this may affect labeling quality")
    
    def log_patches_in_boxes(self, total_patches, patches_in_boxes):
        """Log how many patches are available for labeling."""
        self.logger.info(f"Total patches: {total_patches}")
        self.logger.info(f"Patches in object boxes: {patches_in_boxes}")
        percentage = (patches_in_boxes / total_patches * 100) if total_patches > 0 else 0
        self.logger.info(f"Labeling coverage: {percentage:.1f}%")
    
    def log_grasp_labeling_results(self, good_patches, bad_patches, total_patches):
        """Log the results of grasp labeling."""
        self.logger.info("--- Grasp Labeling Results ---")
        self.logger.info(f"Good grasp patches: {good_patches}")
        self.logger.info(f"Bad grasp patches: {bad_patches}")
        self.logger.info(f"Total labeled patches: {good_patches + bad_patches}")
        if total_patches > 0:
            good_percentage = (good_patches / total_patches * 100)
            self.logger.info(f"Good grasp percentage: {good_percentage:.1f}%")
    
    def log_dataset_creation(self, good_patches, bad_patches, tensor_shape, save_dir):
        """Log dataset creation details."""
        self.logger.info("--- Dataset Creation ---")
        self.logger.info(f"Good patches in dataset: {good_patches}")
        self.logger.info(f"Bad patches in dataset: {bad_patches}")
        self.logger.info(f"Tensor shape: {tensor_shape}")
        self.logger.info(f"Save directory: {save_dir}")
    
    def log_file_save(self, file_path, file_size_mb=None):
        """Log file save operations."""
        self.logger.info(f"Saved: {file_path}")
        if file_size_mb:
            self.logger.info(f"File size: {file_size_mb:.2f} MB")
    
    def log_capture_success(self, capture_index, good_patches, bad_patches):
        """Log successful completion of a capture."""
        self.logger.info(f"✓ Capture {capture_index} completed successfully")
        self.logger.info(f"  Good patches: {good_patches}, Bad patches: {bad_patches}")
    
    def log_capture_error(self, capture_index, folder_name, error):
        """Log capture processing errors."""
        self.logger.error(f"✗ Failed to process capture {capture_index} ({folder_name})")
        self.logger.error(f"Error: {str(error)}")
    
    def log_user_interaction(self, interaction_type, details):
        """Log user interactions during labeling."""
        self.logger.info(f"User interaction - {interaction_type}: {details}")
    
    def log_performance_metrics(self, processing_time, patches_per_second=None):
        """Log performance metrics."""
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        if patches_per_second:
            self.logger.info(f"Processing rate: {patches_per_second:.1f} patches/second")
    
    def log_warning(self, message):
        """Log warnings."""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log errors."""
        self.logger.error(message)
    
    def log_info(self, message):
        """Log general information."""
        self.logger.info(message)
    
    def log_debug(self, message):
        """Log debug information."""
        self.logger.debug(message)
    
    def _get_session_duration(self):
        """Calculate session duration from start timestamp."""
        start_time = datetime.strptime(self.timestamp, "%Y%m%d_%H%M%S")
        end_time = datetime.now()
        duration = end_time - start_time
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def get_log_path(self):
        """Return the path to the log file."""
        return str(self.log_path)
    
    def close(self):
        """Close the logger and all handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def create_logger(dataset_name, log_dir="logs", log_level=logging.INFO):
    """
    Convenience function to create a data labeling logger.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset being labeled
    log_dir : str
        Directory to save log files
    log_level : int
        Logging level
        
    Returns:
    --------
    DataLabelingLogger
        Configured logger instance
    """
    return DataLabelingLogger(dataset_name, log_dir, log_level)