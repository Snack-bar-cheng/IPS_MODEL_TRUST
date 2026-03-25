"""
通用数据集加载器
统一接口加载各种基准数据集
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_benchmark_dataset(dataset_name: str, 
                          test_size: float = 0.2,
                          random_state: int = 42,
                          data_dir: str = "data/benchmark") -> Dict:
    """
    加载基准数据集（统一接口）
    
    参数:
        dataset_name: 数据集名称，格式为 "类型_名称"，如 "uci_boston", "srbench_feynman1"
        test_size: 测试集比例
        random_state: 随机种子
        data_dir: 数据目录
    
    返回:
        dict: 包含以下键的字典
            - 'X_train': 训练特征
            - 'y_train': 训练标签
            - 'X_test': 测试特征
            - 'y_test': 测试标签
            - 'feature_names': 特征名称列表
            - 'target_name': 目标变量名称
            - 'dataset_info': 数据集信息
    """
    dataset_type, name = dataset_name.split('_', 1) if '_' in dataset_name else ('unknown', dataset_name)
    
    if dataset_type == 'uci':
        return load_uci_dataset(name, test_size, random_state, data_dir)
    elif dataset_type == 'srbench':
        return load_srbench_dataset(name, test_size, random_state, data_dir)
    elif dataset_type == 'feynman':
        return load_feynman_dataset(name, test_size, random_state, data_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def list_available_datasets(data_dir: str = "data/benchmark") -> Dict[str, List[str]]:
    """
    列出所有可用的数据集
    
    返回:
        dict: 按类型分组的数据集列表
    """
    datasets = {
        'uci': [],
        'srbench': [],
        'feynman': []
    }
    
    # 列出UCI数据集
    uci_datasets = [
        'boston', 'concrete', 'energy', 'wine', 'airfoil',
        'yacht', 'power_plant', 'bike_sharing'
    ]
    datasets['uci'] = uci_datasets
    
    # 列出SRBench数据集（示例）
    srbench_datasets = [
        'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5',
        'strogatz1', 'strogatz2'
    ]
    datasets['srbench'] = srbench_datasets
    
    # 列出Feynman数据集
    feynman_datasets = [
        'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5',
        'feynman6', 'feynman7', 'feynman8', 'feynman9', 'feynman10'
    ]
    datasets['feynman'] = feynman_datasets
    
    return datasets


def create_data_config_from_benchmark(dataset_dict: Dict, 
                                     output_dir: str = "data/benchmark_configs") -> Dict:
    """
    从基准数据集创建data_config格式
    
    参数:
        dataset_dict: load_benchmark_dataset返回的字典
        output_dir: 输出目录
    
    返回:
        dict: data_config格式的配置
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_info = dataset_dict['dataset_info']
    dataset_name = dataset_info['name']
    
    # 保存训练集和测试集
    train_path = os.path.join(output_dir, f"{dataset_name}_train.csv")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.csv")
    
    # 创建DataFrame并保存
    train_df = pd.DataFrame(
        dataset_dict['X_train'],
        columns=dataset_dict['feature_names']
    )
    train_df[dataset_dict['target_name']] = dataset_dict['y_train']
    train_df.to_csv(train_path, index=False)
    
    test_df = pd.DataFrame(
        dataset_dict['X_test'],
        columns=dataset_dict['feature_names']
    )
    test_df[dataset_dict['target_name']] = dataset_dict['y_test']
    test_df.to_csv(test_path, index=False)
    
    # 创建data_config
    data_config = {
        'all_dataset_path': '',  # 基准数据集可能没有原始文件
        'train_set_path': train_path,
        'test_set_path': test_path,
        'selected_feature': dataset_dict['feature_names'],
        'target_column': dataset_dict['target_name']
    }
    
    # 保存配置
    import json
    config_path = os.path.join(output_dir, f"{dataset_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(data_config, f, indent=2)
    
    logger.info(f"Created data_config for {dataset_name}: {config_path}")
    
    return data_config

