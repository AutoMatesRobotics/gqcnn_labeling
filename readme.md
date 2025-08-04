# GQ-CNN Labeling Tool

A comprehensive interactive tool for manually labeling grasp patches from depth camera data to create training datasets for GQ-CNN (Grasp Quality Convolutional Neural Network) models.

## Overview

This tool provides an end-to-end pipeline for creating labeled grasp datasets from depth camera captures. It features an interactive labeling interface that allows users to manually annotate good and bad grasp locations on depth images, with comprehensive logging and automated dataset generation.

## Features

- **Interactive Labeling Interface**: OpenCV-based GUI for drawing object boundaries and selecting grasp points
- **Patch Extraction**: Automated extraction of depth patches with configurable size and stride
- **Comprehensive Logging**: Detailed session logs with timestamps, user interactions, and performance metrics
- **GQ-CNN Compatible Output**: Generates tensor files compatible with GQ-CNN training pipelines
- **Cross-Platform Support**: Works on Windows and Linux using pathlib for path handling
- **High-Resolution Display**: Configurable window sizes for clear text and visualization
- **Automatic Visualization**: Saves labeled scene visualizations and patch samples

## Repository Structure

```
gqcnn_labeling/
├── data/
│   ├── input/
│   │   └── example_captures/          # Input depth captures
│   │       ├── 2025-06-11_20-51-12/
│   │       │   ├── color_0.png
│   │       │   ├── depth_0.npy        # Required: depth data
│   │       │   └── zivid.intr         # Required: camera intrinsics
│   │       └── 2025-06-30_16-07-26/
│   │           ├── color_0.png
│   │           ├── depth_0.npy
│   │           └── zivid.intr
│   └── output/
│       └── example_dataset/           # Generated dataset files
│           ├── tf_depth_ims_*.npz     # Depth patches tensor
│           ├── grasp_metrics_*.npz    # Quality labels tensor
│           ├── metadata_*.npz         # Dataset metadata
│           └── log/                   # Session logs and visualizations
│               ├── labeled_scene_*.png
│               └── log_*.txt
├── utils/
│   ├── preprocessing.py               # Depth processing and patch extraction
│   ├── labeling.py                   # Interactive labeling interface
│   ├── exporting.py                  # Dataset tensor creation
│   ├── visualising.py                # Visualization functions
│   ├── loading.py                    # Tensor loading utilities
│   └── logger.py                     # Comprehensive logging system
├── main.py                           # Main pipeline script
├── requirements.txt                           # requirements
└── README.md
```

## Installation

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- autolab_core (for camera intrinsics and depth image handling)
- cv2 (OpenCV)

## Installation

1. Create and activate virtualenv:

```bash
cd gqcnn_labeling
python -m venv venv

# Activate the virtual environment:
# On Linux/macOS:
source venv/bin/activate

# On Windows (CMD):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. **Prepare Input Data**: Place a folder of your depth captures in `data/input/` with the required files:
   - `depth_0.npy`: Depth image as numpy array
   - `zivid.intr`: Camera intrinsics file

2. **Run the Labeling Pipeline**:
```bash
python main.py
```

3. **Interactive Labeling Process**:

   **Stage 1: Object Detection**
   - Draw bounding boxes around objects by clicking and dragging
   - Press `c` to clear all boxes
   - Press `SPACE` to proceed to grasp labeling
   - Press `ESC` to exit

   **Stage 2: Grasp Labeling**
   - Left-click to select good grasp points (turn green)
   - Right-click to remove selected points
   - Adjust selection threshold using the slider
   - Press `ESC` when finished

4. **Output**: The tool generates:
   - GQ-CNN compatible tensor files
   - Session logs with detailed metrics
   - Visualization images of labeled scenes

### Configuration

Edit the parameters in `main.py`:

```python
dataset_name = "your_dataset_name"           # Output dataset name
root_dir = Path("data/input/your_captures")  # Input directory
patch_size = 96                              # Patch size (96x96 pixels)
stride = 10                                  # Patch extraction stride (step size)
```

### Advanced Usage

#### Custom Logging
```python
from utils.logger import create_logger

logger = create_logger("custom_dataset", log_dir="custom_logs")
# Then you can use logger throughout your pipeline
logger.log_info("This is an example log record")
```

#### Loading Labeled Tensors
```python
from utils.loading import load_labeled_tensor

images, metrics, metadata = load_labeled_tensor("data/output/dataset", tensor_idx=1)
```

#### Visualizing Results
```python
from utils.visualising import visualize_saved_tensor

visualize_saved_tensor("data/output/dataset", tensor_idx=1, num_samples=8)
```

## File Formats

### Input Files
- **`depth_0.npy`**: NumPy array containing depth values (float32, in meters or millimeters)
- **`zivid.intr`**: Camera intrinsics file (autolab_core format)

### Output Files
- **`tf_depth_ims_*.npz`**: Depth patches tensor (N, H, W, 1)
- **`grasp_metrics_*.npz`**: Binary quality labels (N,) - 1.0 for good, 0.0 for bad
- **`metadata_*.npz`**: Dataset metadata including patch centers, camera intrinsics

## Logging System

The tool provides comprehensive logging of:

- **Session Details**: Start/end times, parameters, duration
- **Processing Metrics**: Patch counts, processing times, success rates
- **User Interactions**: Clicks, box drawings, threshold changes
- **File Operations**: Saves, loads, file sizes
- **Error Handling**: Detailed error messages and warnings

Log files are saved with timestamps: `log_dataset_YYYYMMDD_HHMMSS.txt`