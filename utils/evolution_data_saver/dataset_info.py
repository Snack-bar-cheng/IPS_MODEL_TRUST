"""
数据集信息模块
负责创建和保存数据集信息
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict

from .utils import convert_numpy_types
from .experiment_folder import create_experiment_folder
from . import experiment_folder


def create_dataset_info(data_config, train_features, train_labels, test_features, test_labels, feature_names):
    """
    创建数据集信息结构
    
    参数:
        data_config: 数据配置字典（包含all_dataset_path等信息）
        train_features: 训练特征
        train_labels: 训练标签
        test_features: 测试特征
        test_labels: 测试标签
        feature_names: 特征名称列表
    
    返回:
        dataset_info: 数据集信息字典
    """
    # 读取原始数据文件获取详细信息
    try:
        all_dataset_path = data_config.get('all_dataset_path', '')
        if all_dataset_path and os.path.exists(all_dataset_path):
            data = pd.read_csv(all_dataset_path)
        else:
            data = None
        
        # 基本数据集信息
        dataset_info = {
            "dataset_basic_info": {
                "name": "dataset",
                "file_path": all_dataset_path,
                "file_exists": os.path.exists(all_dataset_path) if all_dataset_path else False,
                "file_size_bytes": os.path.getsize(all_dataset_path) if all_dataset_path and os.path.exists(all_dataset_path) else 0,
                "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "data_structure": {
                "total_samples": len(data) if data is not None else len(train_features) + len(test_features),
                "total_features": len(feature_names),
                "feature_names": feature_names,
                "target_variable": data_config.get('target_column', ['target'])[0] if isinstance(data_config.get('target_column'), list) else data_config.get('target_column', 'target'),
                "train_set_size": len(train_features),
                "test_set_size": len(test_features),
                "train_test_ratio": f"{len(train_features)}:{len(test_features)}"
            },
            "feature_statistics": {},
            "target_statistics": {}
        }
        
        # 特征统计信息
        if data is not None:
            for i, feature_name in enumerate(feature_names):
                if feature_name in data.columns:
                    feature_data = data[feature_name]
                    dataset_info["feature_statistics"][feature_name] = {
                        "index": i,
                        "data_type": str(feature_data.dtype),
                        "missing_values": int(feature_data.isnull().sum()),
                        "unique_values": int(feature_data.nunique()),
                        "statistics": {
                            "mean": float(feature_data.mean()),
                            "std": float(feature_data.std()),
                            "min": float(feature_data.min()),
                            "max": float(feature_data.max()),
                            "25%": float(feature_data.quantile(0.25)),
                            "50%": float(feature_data.quantile(0.50)),
                            "75%": float(feature_data.quantile(0.75))
                        }
                    }
            
            # 目标变量统计信息
            target_col = data_config.get('target_column', [])
            if isinstance(target_col, list) and len(target_col) > 0:
                target_name = target_col[0]
                if target_name in data.columns:
                    target_data = data[target_name]
                    dataset_info["target_statistics"] = {
                        "name": target_name,
                        "data_type": str(target_data.dtype),
                        "missing_values": int(target_data.isnull().sum()),
                        "unique_values": int(target_data.nunique()),
                        "statistics": {
                            "mean": float(target_data.mean()),
                            "std": float(target_data.std()),
                            "min": float(target_data.min()),
                            "max": float(target_data.max()),
                            "25%": float(target_data.quantile(0.25)),
                            "50%": float(target_data.quantile(0.50)),
                            "75%": float(target_data.quantile(0.75))
                        }
                    }
        
        # 训练集和测试集的统计信息
        dataset_info["split_statistics"] = {
            "train_set": {
                "features_shape": list(train_features.shape),
                "labels_shape": list(train_labels.shape),
                "features_stats": {
                    "mean": convert_numpy_types(np.mean(train_features, axis=0)),
                    "std": convert_numpy_types(np.std(train_features, axis=0))
                },
                "labels_stats": {
                    "mean": float(np.mean(train_labels)),
                    "std": float(np.std(train_labels)),
                    "min": float(np.min(train_labels)),
                    "max": float(np.max(train_labels))
                }
            },
            "test_set": {
                "features_shape": list(test_features.shape),
                "labels_shape": list(test_labels.shape),
                "features_stats": {
                    "mean": convert_numpy_types(np.mean(test_features, axis=0)),
                    "std": convert_numpy_types(np.std(test_features, axis=0))
                },
                "labels_stats": {
                    "mean": float(np.mean(test_labels)),
                    "std": float(np.std(test_labels)),
                    "min": float(np.min(test_labels)),
                    "max": float(np.max(test_labels))
                }
            }
        }
        
    except Exception as e:
        # 如果读取失败，返回基本信息
        dataset_info = {
            "dataset_basic_info": {
                "name": "dataset",
                "file_path": data_config.get('all_dataset_path', ''),
                "file_exists": False,
                "error": str(e),
                "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    return dataset_info


def save_dataset_info(dataset_info, experiment_dir, target_name=None):
    """
    保存数据集信息到JSON文件
    
    参数:
        dataset_info: 数据集信息
        experiment_dir: 实验目录
        target_name: 目标变量名称（可选）
    
    返回:
        filepath: 保存的JSON文件路径
    """
    # 保存到evolution_process子文件夹
    evolution_dir = os.path.join(experiment_dir, "evolution_process")
    os.makedirs(evolution_dir, exist_ok=True)
    
    # 保存数据集信息文件（使用安全文件名）
    if target_name:
        filename = f"dataset_info_{target_name}.json"
    else:
        filename = "dataset_info.json"
    filepath = os.path.join(evolution_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    return filepath


def update_dataset_info_with_ridge_formula(dataset_info_path, ridge_formula, train_metrics, test_metrics):
    """
    更新数据集信息JSON文件，添加Ridge公式和最终指标
    
    参数:
        dataset_info_path: 数据集信息JSON文件路径
        ridge_formula: Ridge回归公式
        train_metrics: 训练集指标
        test_metrics: 测试集指标
    """
    try:
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            
            # 添加Ridge公式和最终指标
            dataset_info["gp_final_model"] = {
                "ridge_formula": ridge_formula,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "note": "Train and test metrics show final model performance without cross-validation. But during evolution, 5-fold cross-validation is used to prevent overfitting."
            }
            
            with open(dataset_info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"更新数据集信息失败: {e}")
