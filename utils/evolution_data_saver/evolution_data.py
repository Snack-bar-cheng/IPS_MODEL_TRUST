"""
进化数据保存模块
负责保存和管理进化数据结构
"""

import os
from datetime import datetime
from typing import List, Optional, Dict

from .utils import format_json_compact, convert_to_relative_path
from .experiment_folder import create_experiment_folder
from . import experiment_folder


def create_evolution_data_structure(
    dataset_name, 
    random_seeds, 
    ngen, 
    population_size, 
    config,
    all_dataset_path: Optional[str] = None,
    train_file_path: Optional[str] = None,
    test_file_path: Optional[str] = None,
    train_set_size: Optional[int] = None,
    test_set_size: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    target_variable: Optional[str] = None,
    gp_hyperparams: Optional[Dict] = None
):
    """
    创建进化数据结构（新结构）
    
    参数:
        dataset_name: 数据集名称
        random_seeds: 随机种子
        ngen: 总代数
        population_size: 种群大小
        config: 配置对象
        all_dataset_path: 完整数据集文件路径
        train_file_path: 训练集文件路径
        test_file_path: 测试集文件路径
        train_set_size: 训练集大小
        test_set_size: 测试集大小
        feature_names: 特征名称列表
        target_variable: 目标变量名称
        gp_hyperparams: GP超参数字典
    
    返回:
        evolution_data: 初始化的进化数据结构
    """
    # 构建experiment_info（按指定顺序）
    experiment_info = {
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "random_seeds": random_seeds,
    }
    
    if all_dataset_path:
        experiment_info["all_dataset_path"] = all_dataset_path
    elif config.data_file:
        experiment_info["all_dataset_path"] = config.data_file
    
    if train_file_path:
        experiment_info["train_file_path"] = train_file_path
    if test_file_path:
        experiment_info["test_file_path"] = test_file_path
    if train_set_size is not None:
        experiment_info["train_set_size"] = train_set_size
    if test_set_size is not None:
        experiment_info["test_set_size"] = test_set_size
    if feature_names:
        experiment_info["feature_names"] = feature_names
    if target_variable:
        experiment_info["target_variable"] = target_variable
    
    # 构建gp_info（gp_hyperparameters放在第一位）
    gp_info = {}
    
    # 添加GP超参数（放在第一位）
    if gp_hyperparams:
        gp_hyperparams_with_desc = gp_hyperparams.copy()
        gp_hyperparams_with_desc["description"] = "Genetic Programming hyperparameters used in this experiment"
        gp_info["gp_hyperparameters"] = gp_hyperparams_with_desc
    
    # 添加generations
    gp_info["generations"] = []
    
    evolution_data = {
        "experiment_info": experiment_info,
        "gp_info": gp_info
    }
    
    # 不添加 baseline_info，因为不再需要
    return evolution_data


def finalize_evolution_data(evolution_data):
    """
    完成进化数据记录，添加结束时间（放在start_time后面）
    
    参数:
        evolution_data: 进化数据结构
    """
    # 在start_time后面插入end_time
    exp_info = evolution_data["experiment_info"]
    new_exp_info = {}
    
    # 先添加start_time
    if "start_time" in exp_info:
        new_exp_info["start_time"] = exp_info["start_time"]
    
    # 然后添加end_time
    new_exp_info["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 添加其他字段
    for key, value in exp_info.items():
        if key not in ["start_time", "end_time"]:
            new_exp_info[key] = value
    
    evolution_data["experiment_info"] = new_exp_info


def save_complete_evolution_data(evolution_data, result_dir, dataset_name, random_seeds, target_name=None):
    """
    保存完整的进化数据到单个JSON文件
    
    参数:
        evolution_data: 完整的进化数据
        result_dir: 结果保存目录
        dataset_name: 数据集名称
        random_seeds: 随机种子
    """
    # 使用已创建的实验文件夹
    if experiment_folder._current_experiment_folder is None:
        experiment_folder_path = create_experiment_folder(result_dir, random_seeds, target_name)
    else:
        experiment_folder_path = experiment_folder._current_experiment_folder
    
    # 保存到evolution_process子文件夹
    evolution_dir = os.path.join(experiment_folder_path, "evolution_process")
    
    # 保存到单个JSON文件（以随机种子命名）
    if target_name:
        filename = f"{random_seeds}_{target_name}.json"
    else:
        filename = f"{random_seeds}.json"
    filepath = os.path.join(evolution_dir, filename)
    
    # 重新排序experiment_info字段，并将绝对路径转换为相对路径
    if "experiment_info" in evolution_data:
        exp_info = evolution_data["experiment_info"]
        ordered_exp_info = {}
        # 按指定顺序添加字段
        if "start_time" in exp_info:
            ordered_exp_info["start_time"] = exp_info["start_time"]
        if "end_time" in exp_info:
            ordered_exp_info["end_time"] = exp_info["end_time"]
        if "random_seeds" in exp_info:
            ordered_exp_info["random_seeds"] = exp_info["random_seeds"]
        if "all_dataset_path" in exp_info:
            # 转换为相对路径（相对于JSON文件所在目录）
            ordered_exp_info["all_dataset_path"] = convert_to_relative_path(exp_info["all_dataset_path"], evolution_dir)
        if "train_file_path" in exp_info:
            # 转换为相对路径
            ordered_exp_info["train_file_path"] = convert_to_relative_path(exp_info["train_file_path"], evolution_dir)
        if "test_file_path" in exp_info:
            # 转换为相对路径
            ordered_exp_info["test_file_path"] = convert_to_relative_path(exp_info["test_file_path"], evolution_dir)
        if "train_set_size" in exp_info:
            ordered_exp_info["train_set_size"] = exp_info["train_set_size"]
        if "test_set_size" in exp_info:
            ordered_exp_info["test_set_size"] = exp_info["test_set_size"]
        if "feature_names" in exp_info:
            ordered_exp_info["feature_names"] = exp_info["feature_names"]
        if "target_variable" in exp_info:
            ordered_exp_info["target_variable"] = exp_info["target_variable"]
        evolution_data["experiment_info"] = ordered_exp_info
    
    # 重新排序gp_info字段，确保gp_hyperparameters在第一位
    if "gp_info" in evolution_data:
        gp_info = evolution_data["gp_info"]
        ordered_gp_info = {}
        if "gp_hyperparameters" in gp_info:
            ordered_gp_info["gp_hyperparameters"] = gp_info["gp_hyperparameters"]
        for key, value in gp_info.items():
            if key != "gp_hyperparameters":
                ordered_gp_info[key] = value
        evolution_data["gp_info"] = ordered_gp_info
    
    # 使用自定义格式化函数保存JSON
    json_str = format_json_compact(evolution_data, indent=2)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return experiment_folder_path

