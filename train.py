import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C3, SPPF, Detect, Concat
from types import SimpleNamespace
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoordinateAttention(nn.Module):
    """坐标注意力模块 - 论文实现"""
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = Conv(in_channels, mid_channels, 1, 1)
        self.act = nn.SiLU()
        
        self.conv_h = Conv(mid_channels, out_channels, 1, 1)
        self.conv_w = Conv(mid_channels, out_channels, 1, 1)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        
        x_w = x_w.permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        
        y = self.conv1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        att_h = self.conv_h(x_h).sigmoid()
        att_w = self.conv_w(x_w).sigmoid()
        
        out = identity * att_h * att_w
        return out

class EfficientChannelAttention(nn.Module):
    """高效通道注意力模块 - 论文实现"""
    def __init__(self, in_channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(in_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

class HybridAttentionModule(nn.Module):
    """混合注意力增强模块 (CA + ECA) - 论文核心创新"""
    def __init__(self, in_channels, out_channels):
        super(HybridAttentionModule, self).__init__()
        self.ca = CoordinateAttention(in_channels, out_channels)
        self.eca = EfficientChannelAttention(out_channels)
        
    def forward(self, x):
        x = self.ca(x)
        x = self.eca(x)
        return x

def create_yolov5_config():
    """创建YOLOv5s配置文件"""
    config = {
        'nc': 7,
        'depth_multiple': 0.33,
        'width_multiple': 0.50,
        'anchors': [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ],
        'backbone': [
            [-1, 1, 'Conv', [64, 6, 2, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [-1, 3, 'C3', [128]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [-1, 6, 'C3', [256]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [-1, 9, 'C3', [512]],
            [-1, 1, 'Conv', [1024, 3, 2]],
            [-1, 3, 'C3', [1024]],
            [-1, 1, 'SPPF', [1024, 5]],
        ],
        'head': [
            [-1, 1, 'Conv', [512, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 3, 'C3', [512, False]],
            [-1, 1, 'Conv', [256, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 3, 'C3', [256, False]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 14], 1, 'Concat', [1]],
            [-1, 3, 'C3', [512, False]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],
            [-1, 3, 'C3', [1024, False]],
            [[17, 20, 23], 1, 'Detect', [7]],
        ]
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/yolov5s_radio_tower.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return 'configs/yolov5s_radio_tower.yaml'

def create_ma_yolov5_config():
    """创建MA-YOLOv5配置文件（与YOLOv5结构相同，但会在代码中添加注意力）"""
    return create_yolov5_config()  # 结构相同，注意力在代码中添加

def add_hybrid_attention_to_model(model):
    """为模型添加混合注意力模块 - 在关键位置"""
    logger.info("Adding hybrid attention modules to model...")
    
    # 关键位置索引（根据YOLOv5s结构）
    backbone_end_positions = [9]    # Backbone结束位置（SPPF后）
    neck_positions = [13, 17]       # Neck特征融合后位置
    
    # 添加Backbone结束位置的混合注意力
    for pos in backbone_end_positions:
        if pos < len(model.model):
            module = model.model[pos]
            if hasattr(module, 'conv'):
                in_channels = module.conv.out_channels
                hybrid_att = HybridAttentionModule(in_channels, in_channels)
                module.add_module('hybrid_attention', hybrid_att)
                logger.info(f"Added hybrid attention at backbone position {pos}")
    
    # 添加Neck位置的混合注意力
    for pos in neck_positions:
        if pos < len(model.model):
            module = model.model[pos]
            if hasattr(module, 'cv3'):
                in_channels = module.cv3.conv.out_channels
                hybrid_att = HybridAttentionModule(in_channels, in_channels)
                module.add_module('hybrid_attention', hybrid_att)
                logger.info(f"Added hybrid attention at neck position {pos}")
    
    return model

def create_dataset_config():
    """创建数据集配置文件"""
    data_config = {
        'path': '../datasets/radio_tower',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 7,
        'names': [
            'smoke_fire', 'birds', 'construction_vehicles', 
            'hanging_objects', 'debris_piles', 'wall_collapses', 'bird_nests'
        ]
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/radio_tower.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    return 'data/radio_tower.yaml'

def train_model(model_name, model, config_path, data_path, custom_model=False):
    """训练模型函数"""
    logger.info(f"Starting training for {model_name}...")
    
    train_args = {
        'data': data_path,
        'epochs': 100,  # 为了快速演示，使用100轮，论文是300轮
        'batch': 16,
        'imgsz': 640,
        'device': 0,
        'project': f'runs/train/{model_name}',
        'name': f'{model_name}_exp',
        'optimizer': 'SGD',
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'cos_lr': True,
        'patience': 30,
        'save': True,
        'save_period': 20,
        'cache': True,
        'workers': 4,
        'amp': True,
        'verbose': True,
    }
    
    if custom_model:
        # 对于自定义模型，使用model.train方法
        results = model.train(**train_args)
    else:
        # 对于ultralytics模型，直接训练
        results = model.train(**train_args)
    
    logger.info(f"Training completed for {model_name}")
    return results

def validate_model(model_name, model_path, data_path):
    """验证模型性能"""
    logger.info(f"Validating {model_name}...")
    
    model = YOLO(model_path)
    
    val_results = model.val(
        data=data_path,
        batch=16,
        imgsz=640,
        device=0,
        conf=0.25,
        iou=0.6,
        save_json=True,
        save_hybrid=False,
        half=False,
    )
    
    logger.info(f"Validation completed for {model_name}")
    return val_results

def compare_models():
    """主函数：对比YOLOv5和MA-YOLOv5"""
    
    # 创建配置文件
    yolov5_config = create_yolov5_config()
    ma_yolov5_config = create_ma_yolov5_config()
    data_config = create_dataset_config()
    
    logger.info("Created configuration files:")
    logger.info(f"YOLOv5 config: {yolov5_config}")
    logger.info(f"MA-YOLOv5 config: {ma_yolov5_config}")
    logger.info(f"Data config: {data_config}")
    
    # 创建模型对比实验
    results = {}
    
    try:
        # 实验1：训练基准YOLOv5模型
        logger.info("="*50)
        logger.info("EXPERIMENT 1: Training Baseline YOLOv5")
        logger.info("="*50)
        
        # 加载基准YOLOv5模型
        yolov5_model = YOLO('yolov5s.pt')
        
        # 修改类别数为7
        yolov5_model.model.nc = 7
        yolov5_model.model.names = ['smoke_fire', 'birds', 'construction_vehicles', 
                                   'hanging_objects', 'debris_piles', 'wall_collapses', 'bird_nests']
        
        # 训练YOLOv5
        yolov5_results = train_model('yolov5_baseline', yolov5_model, yolov5_config, data_config)
        results['yolov5_baseline'] = yolov5_results
        
        # 实验2：训练MA-YOLOv5模型
        logger.info("="*50)
        logger.info("EXPERIMENT 2: Training MA-YOLOv5 with Hybrid Attention")
        logger.info("="*50)
        
        # 加载MA-YOLOv5模型（基于YOLOv5s）
        ma_yolov5_model = YOLO('yolov5s.pt')
        
        # 修改类别数
        ma_yolov5_model.model.nc = 7
        ma_yolov5_model.model.names = ['smoke_fire', 'birds', 'construction_vehicles', 
                                      'hanging_objects', 'debris_piles', 'wall_collapses', 'bird_nests']
        
        # 添加混合注意力模块
        ma_yolov5_model = add_hybrid_attention_to_model(ma_yolov5_model)
        
        # 训练MA-YOLOv5
        ma_yolov5_results = train_model('ma_yolov5', ma_yolov5_model, ma_yolov5_config, data_config, custom_model=True)
        results['ma_yolov5'] = ma_yolov5_results
        
        # 实验3：性能验证对比
        logger.info("="*50)
        logger.info("EXPERIMENT 3: Performance Validation Comparison")
        logger.info("="*50)
        
        # 验证YOLOv5
        yolov5_best_model_path = 'runs/train/yolov5_baseline/yolov5_baseline_exp/weights/best.pt'
        if os.path.exists(yolov5_best_model_path):
            yolov5_val = validate_model('YOLOv5_Baseline', yolov5_best_model_path, data_config)
            results['yolov5_validation'] = yolov5_val
        
        # 验证MA-YOLOv5
        ma_yolov5_best_model_path = 'runs/train/ma_yolov5/ma_yolov5_exp/weights/best.pt'
        if os.path.exists(ma_yolov5_best_model_path):
            ma_yolov5_val = validate_model('MA-YOLOv5', ma_yolov5_best_model_path, data_config)
            results['ma_yolov5_validation'] = ma_yolov5_val
        
        # 结果对比分析
        logger.info("="*50)
        logger.info("RESULTS COMPARISON ANALYSIS")
        logger.info("="*50)
        
        if 'yolov5_validation' in results and 'ma_yolov5_validation' in results:
            yolov5_map = results['yolov5_validation'].box.map
            ma_yolov5_map = results['ma_yolov5_validation'].box.map
            
            logger.info(f"YOLOv5 Baseline mAP@0.5: {yolov5_map:.3f}")
            logger.info(f"MA-YOLOv5 mAP@0.5: {ma_yolov5_map:.3f}")
            logger.info(f"Improvement: {ma_yolov5_map - yolov5_map:.3f} ({((ma_yolov5_map - yolov5_map) / yolov5_map * 100):.1f}%)")
            
            # 论文中报告2.2%的mAP提升
            expected_improvement = 0.022
            actual_improvement = (ma_yolov5_map - yolov5_map) / yolov5_map
            
            if actual_improvement >= expected_improvement:
                logger.info("✅ MA-YOLOv5 achieved expected improvement (≥2.2%)")
            else:
                logger.info("⚠️  MA-YOLOv5 improvement below expected (2.2%)")
        
        # 保存实验摘要
        experiment_summary = {
            'experiment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'yolov5_config': yolov5_config,
            'ma_yolov5_config': ma_yolov5_config,
            'data_config': data_config,
            'results': str(results)
        }
        
        with open('experiment_summary.yaml', 'w') as f:
            yaml.dump(experiment_summary, f)
        
        logger.info("Experiment summary saved to experiment_summary.yaml")
        
    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results

def create_sample_dataset_structure():
    """创建示例数据集结构（用于测试）"""
    os.makedirs('datasets/radio_tower/images/train', exist_ok=True)
    os.makedirs('datasets/radio_tower/images/val', exist_ok=True)
    os.makedirs('datasets/radio_tower/labels/train', exist_ok=True)
    os.makedirs('datasets/radio_tower/labels/val', exist_ok=True)
    
    # 创建示例说明文件
    with open('datasets/radio_tower/README.md', 'w') as f:
        f.write("# Radio Tower Inspection Dataset\n")
        f.write("This is a sample dataset structure for radio tower inspection.\n")
        f.write("Please replace with your actual dataset.\n")
        f.write("\n## Structure:\n")
        f.write("- images/train/: Training images\n")
        f.write("- images/val/: Validation images\n")
        f.write("- labels/train/: Training labels (YOLO format)\n")
        f.write("- labels/val/: Validation labels (YOLO format)\n")
    
    logger.info("Created sample dataset structure in datasets/radio_tower/")
    logger.info("Please replace with your actual radio tower inspection dataset")

if __name__ == '__main__':
    # 创建示例数据集结构
    create_sample_dataset_structure()
    
    # 检查是否安装了ultralytics
    try:
        import ultralytics
        logger.info(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        logger.error("Please install ultralytics: pip install ultralytics")
        exit(1)
    
    # 运行对比实验
    logger.info("Starting YOLOv5 vs MA-YOLOv5 comparison experiment...")
    experiment_results = compare_models()
    
    logger.info("Experiment completed!")
    logger.info("Check the 'runs/train/' directory for training results")
    logger.info("Check 'experiment_summary.yaml' for detailed results")