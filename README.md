# MA-YOLOv5: YOLOv5 with Hybrid Attention Modules

## Project Overview

MA-YOLOv5 is an enhanced version of YOLOv5 that incorporates hybrid attention modules (Coordinate Attention + Efficient Channel Attention) for improved object detection performance, specifically designed for radio tower inspection tasks.

## Project Structure

```
ma-yolov5/
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── configs/              # Model configuration files
│   └── yolov5s_radio_tower.yaml
├── data/                 # Dataset configuration files
│   └── radio_tower.yaml
├── datasets/             # Dataset directory
│   └── radio_tower/
│       ├── images/       # Training and validation images
│       │   ├── train/
│       │   └── val/
│       ├── labels/       # YOLO format labels
│       │   ├── train/
│       │   └── val/
│       └── README.md
├── models/               # Model definitions
├── utils/                # Utility functions
└── runs/                 # Training results
    └── train/
```

## Key Features

- **Hybrid Attention Modules**: Combines Coordinate Attention (CA) and Efficient Channel Attention (ECA) for improved feature extraction
- **Enhanced Backbone**: Attention modules integrated into critical positions of the YOLOv5 backbone
- **Custom Dataset Support**: Configured for radio tower inspection with 7 specific classes
- **Comprehensive Experiment Framework**: Includes baseline YOLOv5 comparison

## Classes

The model is trained to detect the following classes for radio tower inspection:

1. `smoke_fire`
2. `birds`
3. `construction_vehicles`
4. `hanging_objects`
5. `debris_piles`
6. `wall_collapses`
7. `bird_nests`

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

#### 2.1 Dataset Source

The dataset can be obtained from Baidu Netdisk:

- **Dataset Name**: tower_detection
- **Link**: https://pan.baidu.com/s/15Ymeg-6NHgloy826UnIg2A
- **Note**: For verification code, please send an email to 3149391417@qq.com

#### 2.2 Dataset Structure

After downloading, extract the dataset to the `datasets/radio_tower/` directory:

```
datasets/radio_tower/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # Training labels (YOLO format)
    └── val/            # Validation labels (YOLO format)
```

### 3. Run Training

Execute the main training script:

```bash
python train.py
```

The script will:
1. Create necessary configuration files
2. Train both baseline YOLOv5 and MA-YOLOv5 models
3. Validate performance and generate comparison results
4. Save results to `experiment_summary.yaml`

### 4. Training Parameters

Key training parameters can be modified in the `train_model` function:

- `epochs`: Number of training epochs (default: 100)
- `batch`: Batch size (default: 16)
- `imgsz`: Image size (default: 640)
- `optimizer`: Optimizer choice (default: SGD)
- `lr0`: Initial learning rate (default: 0.01)

## Expected Results

MA-YOLOv5 is expected to achieve approximately 2.2% mAP improvement over baseline YOLOv5, as reported in the accompanying research paper.

## Model Architecture

The hybrid attention module is integrated at key positions:
- After the SPPF layer in the backbone
- After feature fusion nodes in the neck


## Citation

If you use this code in your research, please cite our paper:

```
@article{MA-YOLOv52026,
  title={MA-YOLOv5: Enhanced YOLOv5 with Hybrid Attention for Radio Tower Inspection},
  author={Ni zhengqi},
  journal={springer nature},
  year={2026}
}
```