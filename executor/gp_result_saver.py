"""
GP结果保存模块
"""

import os
import json
from typing import Optional

# 导入参考实现
import sys
reference_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gps_lr_llm_step_residual')
if reference_path not in sys.path:
    sys.path.insert(0, reference_path)

# 导入本地可视化模块 - 先设置路径，再导入
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from evolution_data_saver.utils import format_json_compact
from evolution_visualization import Plot_tree


def save_results(evolution_data: dict, target_dir: str, target_name: str,
                random_seed: int, timestamp: str, best_individual, 
                X_train, y_train, X_test, y_test, feature_names: list,
                llm_features=None) -> Optional[str]:
    """
    保存GP结果
    
    参数:
        evolution_data: 进化数据
        target_dir: 目标目录
        target_name: 目标变量名称
        random_seed: 随机种子
        timestamp: 时间戳
        best_individual: 最佳个体（从种群中获取）
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        feature_names: 特征名称列表
        llm_features: LLM特征列表（可选），用于可视化比对
    
    返回:
        str: 保存的文件路径
    """
    filename = f"gps_{random_seed}_{timestamp}.json"
    filepath = os.path.join(target_dir, filename)
    
    # 保存进化数据（不包含baseline_info）
    result_data = {
        "experiment_info": evolution_data["experiment_info"],
        "gp_info": evolution_data["gp_info"]
    }
    
    # 使用参考实现的格式化函数
    json_str = format_json_compact(result_data, indent=2)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    # 保存树可视化
    if best_individual is not None:
        tree_save_dir = os.path.join(target_dir, 'best_tree_visualization')
        os.makedirs(tree_save_dir, exist_ok=True)
        
        try:
            # 从evolution_data中提取ridge_formula（如果存在）
            ridge_formula = None
            if evolution_data and "gp_info" in evolution_data:
                gp_info = evolution_data["gp_info"]
                if "final_model" in gp_info and "ridge_formula" in gp_info["final_model"]:
                    ridge_formula = gp_info["final_model"]["ridge_formula"]
            
            # 调用可视化函数（最后一代，is_last_generation=True）
            Plot_tree(
                best_individual, random_seed, tree_save_dir, 
                feature_names, 0,  # generation_i=0表示最后一代
                target_name=target_name, 
                llm_features=llm_features, 
                is_last_generation=True,
                ridge_formula=ridge_formula
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"树可视化失败: {e}")
    
    return filepath

