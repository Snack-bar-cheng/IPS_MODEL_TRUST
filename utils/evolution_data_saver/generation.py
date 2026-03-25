"""
代信息保存模块
负责保存每一代的进化信息
"""

import numpy as np
from datetime import datetime
from typing import List, Optional

from .utils import convert_numpy_types
from .feature_analysis import count_feature_usage, generate_ridge_formula_for_individual


def save_generation_info(gen, population, halloffame, stats_record, evolution_data, 
                        toolbox=None, train_features=None, train_labels=None, 
                        feature_names=None, target_variable=None, enable_high=True,
                        cumulative_time_seconds=None):
    """
    将当前代信息添加到进化数据中（新结构）
    
    参数:
        gen: 当前代数
        population: 当前种群
        halloffame: 名人堂
        stats_record: 统计记录
        evolution_data: 完整的进化数据字典（新结构）
        toolbox: DEAP工具箱（用于生成ridge_formula，可选）
        train_features: 训练特征数据（用于生成ridge_formula，可选）
        train_labels: 训练标签数据（用于生成ridge_formula，可选）
        feature_names: 特征名称列表（用于生成feature_usage，可选）
        target_variable: 目标变量名称（用于生成ridge_formula，可选）
        enable_high: 是否启用High函数模式，默认True
        cumulative_time_seconds: 累计训练时间（秒），从开始到当前代结束的累计时间
    """
    # 收集当前代信息（字段顺序：generation, timestamp, cumulative_time_seconds, statistics, whole_individuals）
    generation_info = {
        "generation": gen,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "statistics": convert_numpy_types(stats_record),
        "whole_individuals": []
    }
    
    # 添加累计时间字段（放在timestamp之后）
    if cumulative_time_seconds is not None:
        # 在timestamp之后插入cumulative_time_seconds字段
        # 由于Python 3.7+字典保持插入顺序，我们需要重新构建字典
        generation_info = {
            "generation": gen,
            "timestamp": generation_info["timestamp"],
            "cumulative_time_seconds": float(cumulative_time_seconds),
            "statistics": generation_info["statistics"],
            "whole_individuals": generation_info["whole_individuals"]
        }
    
    # 获取特征名称和目标变量（如果未提供，尝试从evolution_data获取）
    if feature_names is None:
        feature_names = evolution_data.get("experiment_info", {}).get("feature_names", [])
    if target_variable is None:
        target_variable = evolution_data.get("experiment_info", {}).get("target_variable", "y")
    
    # 对种群按适应度排序（降序，适应度高的在前）
    sorted_population = sorted(
        population, 
        key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf,
        reverse=True
    )
    
    # 记录所有个体信息
    for rank, ind in enumerate(sorted_population, start=1):
        # 获取CV指标 (High函数模式)
        ind_cv_metrics = getattr(ind, "cv_metrics", None)
        mse_val = None
        rmse_val = None
        mae_val = None
        
        if ind_cv_metrics and isinstance(ind_cv_metrics, dict):
            # 新的cv_metrics格式: {folds: 5, cross_validation: {...}}
            cross_validation_summary = ind_cv_metrics.get("cross_validation", {})
            if cross_validation_summary:
                mse_val = cross_validation_summary.get("mse_mean")
                rmse_val = cross_validation_summary.get("rmse_mean")
                mae_val = cross_validation_summary.get("mae_mean")
            else:
                # 兼容旧的cv_metrics格式: {folds: [...], summary: {...}}
                summary = ind_cv_metrics.get("summary", {})
                mse_val = summary.get("mse_mean")
                rmse_val = summary.get("rmse_mean")
                mae_val = summary.get("mae_mean")
        
        # 如果是传统GP模式，从train_metrics获取
        if mse_val is None:
            ind_train_metrics = getattr(ind, "train_metrics", None)
            if ind_train_metrics and isinstance(ind_train_metrics, dict):
                mse_val = ind_train_metrics.get("mse")
                rmse_val = ind_train_metrics.get("rmse")
                mae_val = ind_train_metrics.get("mae")
        
        # 获取GP表达式
        gp_expression = str(ind)
        
        # 生成Ridge公式（如果提供了必要的参数且启用了High函数）
        ridge_formula = None
        if enable_high and toolbox is not None and train_features is not None and train_labels is not None:
            ridge_formula = generate_ridge_formula_for_individual(
                ind, toolbox, train_features, train_labels, target_variable, enable_high=enable_high
            )
        
        # 统计特征使用次数
        feature_usage = count_feature_usage(gp_expression, feature_names) if feature_names else {}
        
        # 安全地获取height，避免空表达式导致的错误
        try:
            if len(ind) > 0:
                height = int(ind.height)
            else:
                height = 0
        except (IndexError, AttributeError):
            # 如果计算height失败（可能是空表达式），使用0作为默认值
            height = 0
        
        # 构建个体信息
        individual_info = {
            "rank": rank,
            "gp_expression": gp_expression,
            "ridge_formula": ridge_formula,
            "fitness": float(ind.fitness.values[0]) if ind.fitness.valid else None,
            "mse": float(mse_val) if mse_val is not None else None,
            "rmse": float(rmse_val) if rmse_val is not None else None,
            "mae": float(mae_val) if mae_val is not None else None,
            "size": int(len(ind)),
            "height": height,
            "feature_usage": feature_usage
        }
        
        generation_info["whole_individuals"].append(individual_info)
    
    # 添加到gp_info.generations中
    if "gp_info" not in evolution_data:
        evolution_data["gp_info"] = {"generations": []}
    if "generations" not in evolution_data["gp_info"]:
        evolution_data["gp_info"]["generations"] = []
    
    evolution_data["gp_info"]["generations"].append(generation_info)

