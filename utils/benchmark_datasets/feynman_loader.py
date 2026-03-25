"""
Feynman数据集加载器
专门用于可解释性数据集
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Feynman公式的ground truth表达式（示例）
FEYNMAN_GROUND_TRUTH = {
    'feynman1': 'I = (m1 * m2) / ((m1 + m2) * r^2)',
    'feynman2': 'E = (1/2) * m * v^2',
    'feynman3': 'F = G * (m1 * m2) / r^2',
    # 可以添加更多
}


def load_feynman_dataset(dataset_name: str,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        data_dir: str = "data/benchmark/feynman") -> Dict:
    """
    加载Feynman数据集（可解释性数据集）
    
    参数:
        dataset_name: 数据集名称（如 'feynman1', 'feynman2'）
        test_size: 测试集比例
        random_state: 随机种子
        data_dir: 数据目录
    
    返回:
        dict: 包含X_train, y_train, X_test, y_test, feature_names, target_name, ground_truth, dataset_info
    """
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(file_path):
        logger.warning(f"Feynman dataset {dataset_name} not found at {file_path}")
        logger.info("Please download Feynman datasets from:")
        logger.info("  - https://space.mit.edu/home/tegmark/aifeynman.html")
        logger.info("  - Or from SRBench: https://github.com/cavalab/srbench")
        raise FileNotFoundError(f"Please download Feynman dataset {dataset_name} to {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Feynman格式：最后一列是目标变量，其他是特征
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]
    
    X = df[feature_names].values
    y = df[target_name].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 获取ground truth表达式
    ground_truth = FEYNMAN_GROUND_TRUTH.get(dataset_name, None)
    
    # 尝试从文件加载ground truth
    ground_truth_path = os.path.join(data_dir, f"{dataset_name}_ground_truth.txt")
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
        'ground_truth': ground_truth,  # 关键：用于可解释性评估
        'dataset_info': {
            'name': dataset_name,
            'type': 'feynman',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': f'Feynman {dataset_name}',
            'has_ground_truth': ground_truth is not None,
            'ground_truth_expr': ground_truth
        }
    }

