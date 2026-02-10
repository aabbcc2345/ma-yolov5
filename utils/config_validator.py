import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_dataset_config(config_path):
    """验证数据集配置文件"""
    logger.info(f"Validating dataset configuration: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        errors = []
        warnings = []
        
        # 检查必需字段
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # 检查路径
        if 'path' in config:
            dataset_path = Path(config['path'])
            if not dataset_path.exists():
                errors.append(f"Dataset path does not exist: {dataset_path}")
            
            # 检查训练和验证路径
            for split in ['train', 'val']:
                if split in config:
                    split_path = dataset_path / config[split]
                    if not split_path.exists():
                        warnings.append(f"{split} path may not exist: {split_path}")
        
        # 检查类别数量
        if 'nc' in config and 'names' in config:
            if len(config['names']) != config['nc']:
                errors.append(f"Class count mismatch: names={len(config['names'])}, nc={config['nc']}")
        
        # 检查类别名称
        if 'names' in config:
            if not isinstance(config['names'], list):
                errors.append("'names' must be a list")
            elif len(config['names']) == 0:
                errors.append("'names' list is empty")
        
        # 输出结果
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        if warnings:
            logger.warning("Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("✓ Dataset configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

def validate_model_config(config_path):
    """验证模型配置文件"""
    logger.info(f"Validating model configuration: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        errors = []
        
        # 检查必需字段
        required_fields = ['nc', 'depth_multiple', 'width_multiple', 'backbone', 'head']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # 检查backbone和head结构
        if 'backbone' in config:
            if not isinstance(config['backbone'], list):
                errors.append("'backbone' must be a list")
        
        if 'head' in config:
            if not isinstance(config['head'], list):
                errors.append("'head' must be a list")
        
        # 检查nc值
        if 'nc' in config:
            if not isinstance(config['nc'], int) or config['nc'] <= 0:
                errors.append(f"'nc' must be a positive integer, got: {config['nc']}")
        
        # 检查depth_multiple和width_multiple
        for field in ['depth_multiple', 'width_multiple']:
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"'{field}' must be a positive number, got: {value}")
        
        # 输出结果
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✓ Model configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

def validate_all_configs():
    """验证所有配置文件"""
    logger.info("=" * 50)
    logger.info("Validating All Configuration Files")
    logger.info("=" * 50)
    
    results = {}
    
    # 验证数据集配置
    dataset_config = 'data/radio_tower.yaml'
    if os.path.exists(dataset_config):
        results['dataset'] = validate_dataset_config(dataset_config)
    else:
        logger.warning(f"Dataset config not found: {dataset_config}")
        results['dataset'] = False
    
    # 验证模型配置
    model_configs = [
        'configs/yolov5s_radio_tower.yaml',
        'configs/ma_yolov5_radio_tower.yaml'
    ]
    
    for model_config in model_configs:
        if os.path.exists(model_config):
            config_name = os.path.basename(model_config)
            results[config_name] = validate_model_config(model_config)
        else:
            logger.warning(f"Model config not found: {model_config}")
            results[os.path.basename(model_config)] = False
    
    # 总结
    logger.info("=" * 50)
    logger.info("Validation Summary")
    logger.info("=" * 50)
    
    for config_name, is_valid in results.items():
        status = "✓ VALID" if is_valid else "✗ INVALID"
        logger.info(f"{config_name}: {status}")
    
    all_valid = all(results.values())
    if all_valid:
        logger.info("✓ All configurations are valid!")
    else:
        logger.warning("⚠ Some configurations have issues")
    
    return all_valid

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    validate_all_configs()