"""
实验文件夹管理模块
负责创建和管理实验文件夹
"""

import os
from typing import Union, List, Optional


# 全局变量：当前实验文件夹路径
_current_experiment_folder = None


def create_experiment_folder(result_dir: str, random_seed: Union[int, List[int]], target_name: Optional[str] = None) -> str:
    """
    创建实验文件夹，包含子文件夹
    
    参数:
        result_dir: 结果保存目录
        random_seed: 随机种子（单个整数或列表，不再用于创建子目录）
        target_name: 目标变量名称（可选）
    
    返回:
        str: 实验文件夹路径（直接使用result_dir/target_name，不创建随机种子子目录）
    """
    global _current_experiment_folder
    
    # 直接使用提供的result目录作为实验目录（不创建随机种子子目录）
    # 将目标加入命名空间目录中，避免不同目标混写不清
    if target_name:
        experiment_folder_path = os.path.join(result_dir, target_name)
    else:
        experiment_folder_path = result_dir
    os.makedirs(experiment_folder_path, exist_ok=True)
    
    # 创建子文件夹
    evolution_process_dir = os.path.join(experiment_folder_path, "evolution_process")
    best_tree_dir = os.path.join(experiment_folder_path, "best_tree_visualization")
    
    os.makedirs(evolution_process_dir, exist_ok=True)
    os.makedirs(best_tree_dir, exist_ok=True)
    
    # 保存当前实验文件夹路径
    _current_experiment_folder = experiment_folder_path
    
    return experiment_folder_path


def get_current_experiment_folder() -> Optional[str]:
    """
    获取当前实验文件夹路径
    
    返回:
        Optional[str]: 当前实验文件夹路径，如果未设置则返回None
    """
    return _current_experiment_folder


def reset_experiment_folder():
    """重置当前实验文件夹"""
    global _current_experiment_folder
    _current_experiment_folder = None

