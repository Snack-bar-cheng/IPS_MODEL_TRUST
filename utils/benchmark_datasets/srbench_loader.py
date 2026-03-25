"""
SRBench数据集加载器
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


def load_srbench_dataset(dataset_name: str,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         data_dir: str = "data/benchmark/srbench") -> Dict:
    """
    加载SRBench数据集
    
    参数:
        dataset_name: 数据集名称（如 'feynman1', 'strogatz1'）
        test_size: 测试集比例
        random_state: 随机种子
        data_dir: 数据目录
    
    返回:
        dict: 包含X_train, y_train, X_test, y_test, feature_names, target_name, dataset_info
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # SRBench数据集通常存储在特定格式中
    # 这里提供框架，实际需要根据SRBench的具体格式调整
    
    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(file_path):
        logger.warning(f"SRBench dataset {dataset_name} not found at {file_path}")
        logger.info("Please download SRBench datasets from: https://github.com/cavalab/srbench")
        logger.info("Or use the script: python scripts/download_srbench.py")
        raise FileNotFoundError(f"Please download SRBench dataset {dataset_name} to {file_path}")
    
    df = pd.read_csv(file_path)
    
    # SRBench格式：最后一列是目标变量，其他是特征
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]
    
    X = df[feature_names].values
    y = df[target_name].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 尝试加载ground truth表达式（如果有）
    ground_truth_path = os.path.join(data_dir, f"{dataset_name}_ground_truth.txt")
    ground_truth = None
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r') as f:
            ground_truth = f.read().strip()
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_name': target_name,
        'ground_truth': ground_truth,  # 可解释性数据集的关键
        'dataset_info': {
            'name': dataset_name,
            'type': 'srbench',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': f'SRBench {dataset_name}',
            'has_ground_truth': ground_truth is not None
        }
    }

