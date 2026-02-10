# Model Configuration Files

This directory contains configuration files for different YOLOv5 model variants.

## Configuration Files

### 1. `yolov5s_radio_tower.yaml`
- **Purpose**: Baseline YOLOv5s model configuration for radio tower inspection
- **Classes**: 7 classes (smoke_fire, birds, construction_vehicles, hanging_objects, debris_piles, wall_collapses, bird_nests)
- **Architecture**: Standard YOLOv5s backbone and head
- **Usage**: Used for baseline experiments and comparison

### 2. `ma_yolov5_radio_tower.yaml`
- **Purpose**: MA-YOLOv5 model configuration with hybrid attention modules
- **Classes**: 7 classes (same as baseline)
- **Architecture**: YOLOv5s backbone and head with hybrid attention modules
- **Attention Modules**:
  - Coordinate Attention (CA)
  - Efficient Channel Attention (ECA)
- **Integration Positions**:
  - Position 9: After SPPF layer (backbone end)
  - Position 13: After first feature fusion (neck)
  - Position 17: After second feature fusion (neck)
- **Usage**: Used for enhanced model experiments

## Configuration Parameters

### Model Parameters
- `nc`: Number of classes (7)
- `depth_multiple`: Depth multiplier (0.33 for YOLOv5s)
- `width_multiple`: Width multiplier (0.5 for YOLOv5s)
- `anchors`: Anchor boxes for different scales

### Attention Parameters (MA-YOLOv5 only)
- `coordinate_attention.reduction`: Channel reduction ratio (32)
- `efficient_channel_attention.gamma`: Gamma parameter for kernel size calculation (2)
- `efficient_channel_attention.b`: Beta parameter for kernel size calculation (1)

## Architecture Details

### Backbone
- Conv layers with different kernel sizes and strides
- C3 (Cross Stage Partial) blocks for feature extraction
- SPPF (Spatial Pyramid Pooling - Fast) for multi-scale features

### Head
- Feature pyramid network (FPN) structure
- Upsampling and concatenation for multi-scale detection
- Detect layer for final predictions

## Usage

### Loading Configuration in Training Script

```python
from train import create_yolov5_config, create_ma_yolov5_config

# Create baseline configuration
yolov5_config = create_yolov5_config()
# Output: configs/yolov5s_radio_tower.yaml

# Create MA-YOLOv5 configuration
ma_yolov5_config = create_ma_yolov5_config()
# Output: configs/ma_yolov5_radio_tower.yaml
```

### Manual Configuration Loading

```python
import yaml

# Load configuration
with open('configs/yolov5s_radio_tower.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
num_classes = config['nc']
depth_mult = config['depth_multiple']
```

## Customization

### Adding New Classes
Modify the `nc` parameter and update the last layer in head:
```yaml
nc: 10  # Change from 7 to 10
# Update last line in head:
# [[17, 20, 23], 1, Detect, [10]]  # Change from 7 to 10
```

### Adjusting Model Size
- **YOLOv5n**: depth_multiple=0.33, width_multiple=0.25
- **YOLOv5s**: depth_multiple=0.33, width_multiple=0.50
- **YOLOv5m**: depth_multiple=0.67, width_multiple=0.75
- **YOLOv5l**: depth_multiple=1.00, width_multiple=1.00
- **YOLOv5x**: depth_multiple=1.33, width_multiple=1.25

### Modifying Attention Parameters
In `ma_yolov5_radio_tower.yaml`:
```yaml
attention:
  coordinate_attention:
    reduction: 16  # Smaller reduction = more parameters
  efficient_channel_attention:
    gamma: 3      # Larger gamma = smaller kernel
    b: 1
```

## Notes

- Configuration files are automatically created by the training script
- MA-YOLOv5 uses the same base architecture as YOLOv5s
- Attention modules are dynamically added during model initialization
- Ensure configuration parameters match the dataset configuration in `data/radio_tower.yaml`