# Dataset Configuration Files

This directory contains dataset configuration files for training and evaluation.

## Configuration Files

### `radio_tower.yaml`
Main dataset configuration file for radio tower inspection task.

## Configuration Structure

```yaml
path: ../datasets/radio_tower    # Root path to dataset
train: images/train              # Training images path (relative to path)
val: images/val                  # Validation images path (relative to path)
test: images/test                # Test images path (relative to path)
nc: 7                           # Number of classes
names:                          # Class names
  - smoke_fire
  - birds
  - construction_vehicles
  - hanging_objects
  - debris_piles
  - wall_collapses
  - bird_nests
```

## Parameters Explained

### `path`
- **Description**: Root directory path to the dataset
- **Format**: Relative path from project root or absolute path
- **Example**: `../datasets/radio_tower` or `/path/to/datasets/radio_tower`

### `train`, `val`, `test`
- **Description**: Paths to training, validation, and test images
- **Format**: Relative paths to `path`
- **Note**: Paths should point to image directories, not label directories

### `nc`
- **Description**: Number of classes in the dataset
- **Type**: Integer
- **Example**: 7 for radio tower inspection dataset

### `names`
- **Description**: List of class names
- **Format**: List of strings
- **Order**: Must match the class IDs in label files (0 to nc-1)

## Usage

### In Training Script

```python
from train import create_dataset_config

# Create dataset configuration
data_config = create_dataset_config()
# Output: data/radio_tower.yaml

# Use in training
train_args = {
    'data': data_config,
    'epochs': 100,
    'batch': 16,
    # ... other parameters
}
```

### Manual Loading

```python
import yaml

# Load configuration
with open('data/radio_tower.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Access parameters
dataset_path = data_config['path']
train_path = data_config['train']
num_classes = data_config['nc']
class_names = data_config['names']
```

### With Ultralytics YOLO

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov5s.pt')

# Train with dataset configuration
results = model.train(
    data='data/radio_tower.yaml',
    epochs=100,
    batch=16
)
```

## Class Mapping

The class names in `names` correspond to class IDs in YOLO format labels:

| Class ID | Class Name              | Description                          |
|----------|------------------------|--------------------------------------|
| 0        | smoke_fire            | Smoke and fire detection              |
| 1        | birds                | Birds on or near tower               |
| 2        | construction_vehicles | Construction vehicles around tower    |
| 3        | hanging_objects      | Objects hanging from tower           |
| 4        | debris_piles         | Debris piles at base                 |
| 5        | wall_collapses       | Wall collapse areas                   |
| 6        | bird_nests           | Bird nests on tower                  |

## Dataset Requirements

### Image Format
- Supported formats: JPG, JPEG, PNG, BMP, WEBP
- Recommended size: 640x640 (will be resized during training)
- Color space: RGB (3 channels)

### Label Format
- Format: YOLO format (`.txt` files)
- Structure: One line per object
- Line format: `class_id center_x center_y width height`
- Coordinates: Normalized to [0, 1]

### Directory Structure
```
datasets/radio_tower/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images (optional)
└── labels/
    ├── train/          # Training labels
    ├── val/            # Validation labels
    └── test/           # Test labels (optional)
```

## Customization

### Adding New Classes
```yaml
nc: 8  # Increase from 7 to 8
names:
  - smoke_fire
  - birds
  - construction_vehicles
  - hanging_objects
  - debris_piles
  - wall_collapses
  - bird_nests
  - new_class_name  # Add new class
```

### Changing Dataset Path
```yaml
path: /absolute/path/to/your/dataset  # Use absolute path
# or
path: ../../other/location/dataset     # Use different relative path
```

### Modifying Data Splits
```yaml
train: images/train  # Change to different directory
val: images/validation  # Rename validation directory
test: images/test_set  # Rename test directory
```

## Validation

### Check Configuration Validity
```python
import yaml
import os

# Load configuration
with open('data/radio_tower.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Validate paths
dataset_path = config['path']
for split in ['train', 'val', 'test']:
    split_path = os.path.join(dataset_path, config[split])
    if os.path.exists(split_path):
        print(f"✓ {split} path exists: {split_path}")
    else:
        print(f"✗ {split} path missing: {split_path}")

# Validate class count
if len(config['names']) == config['nc']:
    print(f"✓ Class count matches: {config['nc']}")
else:
    print(f"✗ Class count mismatch: names={len(config['names'])}, nc={config['nc']}")
```

## Notes

- Ensure configuration file matches the actual dataset structure
- Class names must match the order in model configuration
- Paths can be relative or absolute
- Test split is optional but recommended for final evaluation
- Keep configuration file in version control for reproducibility