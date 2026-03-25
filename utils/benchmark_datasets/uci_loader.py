"""
UCI数据集加载器
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# UCI数据集下载URL（需要手动下载的数据集）
UCI_DATASETS = {
    'boston': {
        'loader': 'sklearn',
        'name': 'Boston Housing',
        'description': 'Predicting house prices'
    },
    'concrete': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
        'name': 'Concrete Compressive Strength',
        'description': 'Predicting concrete strength'
    },
    'energy': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
        'name': 'Energy Efficiency',
        'description': 'Predicting building energy consumption'
    },
    'wine': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'name': 'Wine Quality',
        'description': 'Predicting wine quality'
    },
    'airfoil': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat',
        'name': 'Airfoil Self-Noise',
        'description': 'Predicting airfoil noise'
    }
}


def load_uci_dataset(dataset_name: str,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    data_dir: str = "data/benchmark/uci") -> Dict:
    """
    加载UCI数据集
    
    参数:
        dataset_name: 数据集名称（如 'boston', 'concrete'）
        test_size: 测试集比例
        random_state: 随机种子
        data_dir: 数据目录
    
    返回:
        dict: 包含X_train, y_train, X_test, y_test, feature_names, target_name, dataset_info
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == 'boston':
        return _load_boston(test_size, random_state)
    elif dataset_name == 'concrete':
        return _load_concrete(data_dir, test_size, random_state)
    elif dataset_name == 'energy':
        return _load_energy(data_dir, test_size, random_state)
    elif dataset_name == 'wine':
        return _load_wine(data_dir, test_size, random_state)
    elif dataset_name == 'airfoil':
        return _load_airfoil(data_dir, test_size, random_state)
    else:
        raise ValueError(f"Unknown UCI dataset: {dataset_name}")


def _load_boston(test_size: float, random_state: int) -> Dict:
    """加载Boston Housing数据集（使用sklearn）"""
    try:
        data = load_boston()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_name = 'MEDV'
    except:
        # sklearn新版本可能移除了boston数据集，使用替代方法
        logger.warning("Boston dataset not available in sklearn, using alternative")
        # 这里可以添加从UCI直接下载的代码
        raise NotImplementedError("Please download Boston dataset manually from UCI")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_name': target_name,
        'dataset_info': {
            'name': 'boston',
            'type': 'uci',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': 'Boston Housing Prices'
        }
    }


def _load_concrete(data_dir: str, test_size: float, random_state: int) -> Dict:
    """加载Concrete数据集"""
    file_path = os.path.join(data_dir, 'Concrete_Data.csv')
    
    if not os.path.exists(file_path):
        logger.warning(f"Concrete dataset not found at {file_path}")
        logger.info("Please download from: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength")
        raise FileNotFoundError(f"Please download Concrete dataset to {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 假设最后一列是目标变量
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]
    
    X = df[feature_names].values
    y = df[target_name].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_name': target_name,
        'dataset_info': {
            'name': 'concrete',
            'type': 'uci',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': 'Concrete Compressive Strength'
        }
    }


def _load_energy(data_dir: str, test_size: float, random_state: int) -> Dict:
    """加载Energy Efficiency数据集"""
    file_path = os.path.join(data_dir, 'ENB2012_data.xlsx')
    
    if not os.path.exists(file_path):
        logger.warning(f"Energy dataset not found at {file_path}")
        logger.info("Please download from: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency")
        raise FileNotFoundError(f"Please download Energy dataset to {file_path}")
    
    df = pd.read_excel(file_path)
    
    # Energy数据集有两个目标变量，选择第一个（Heating Load）
    feature_names = df.columns[:-2].tolist()
    target_name = df.columns[-2]  # Heating Load
    
    X = df[feature_names].values
    y = df[target_name].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_name': target_name,
        'dataset_info': {
            'name': 'energy',
            'type': 'uci',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': 'Energy Efficiency'
        }
    }


def _load_wine(data_dir: str, test_size: float, random_state: int) -> Dict:
    """加载Wine Quality数据集"""
    file_path = os.path.join(data_dir, 'winequality-red.csv')
    
    if not os.path.exists(file_path):
        logger.warning(f"Wine dataset not found at {file_path}")
        logger.info("Please download from: https://archive.ics.uci.edu/ml/datasets/Wine+Quality")
        raise FileNotFoundError(f"Please download Wine dataset to {file_path}")
    
    df = pd.read_csv(file_path, sep=';')
    
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]
    
    X = df[feature_names].values
    y = df[target_name].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_name': target_name,
        'dataset_info': {
            'name': 'wine',
            'type': 'uci',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': 'Wine Quality'
        }
    }


def _load_airfoil(data_dir: str, test_size: float, random_state: int) -> Dict:
    """加载Airfoil Self-Noise数据集"""
    file_path = os.path.join(data_dir, 'airfoil_self_noise.dat')
    
    if not os.path.exists(file_path):
        logger.warning(f"Airfoil dataset not found at {file_path}")
        logger.info("Please download from: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise")
        raise FileNotFoundError(f"Please download Airfoil dataset to {file_path}")
    
    # Airfoil数据集是空格分隔的
    df = pd.read_csv(file_path, sep='\s+', header=None)
    
    feature_names = [f'X{i+1}' for i in range(df.shape[1] - 1)]
    target_name = 'y'
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_name': target_name,
        'dataset_info': {
            'name': 'airfoil',
            'type': 'uci',
            'n_samples': len(X),
            'n_features': len(feature_names),
            'description': 'Airfoil Self-Noise'
        }
    }

