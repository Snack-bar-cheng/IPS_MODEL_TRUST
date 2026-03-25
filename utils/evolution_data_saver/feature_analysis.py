"""
特征分析模块
包含特征使用统计、Ridge公式生成等功能
"""

import numpy as np
import re
from typing import List, Dict


def count_feature_usage(expression: str, feature_names: List[str]) -> Dict[str, int]:
    """
    统计表达式中每个特征的使用次数
    
    参数:
        expression: GP表达式字符串
        feature_names: 特征名称列表
    
    返回:
        feature_usage: 特征使用次数字典 {feature_name: count}
    """
    feature_usage = {name: 0 for name in feature_names}
    
    # 使用正则表达式匹配特征名称（作为完整单词，避免部分匹配）
    for feature_name in feature_names:
        # 使用单词边界匹配，确保匹配完整的特征名
        # 例如 "SiO2" 不应该匹配到 "SiO2_Dry" 的一部分
        pattern = r'\b' + re.escape(feature_name) + r'\b'
        matches = re.findall(pattern, expression)
        feature_usage[feature_name] = len(matches)
    
    return feature_usage


def generate_ridge_formula_for_individual(individual, toolbox, train_features, train_labels, target_variable='y', enable_high=True):
    """
    为单个个体生成Ridge回归公式
    
    参数:
        individual: GP个体
        toolbox: DEAP工具箱
        train_features: 训练特征数据
        train_labels: 训练标签数据
        target_variable: 目标变量名称
        enable_high: 是否启用High函数模式，默认True
    
    返回:
        ridge_formula: Ridge回归公式字符串，如果生成失败则返回None
    """
    try:
        from sklearn.linear_model import RidgeCV
        # 导入路径修复：与项目其他部分保持一致的方式
        # 确保 utils 目录在 sys.path 中，以便导入 gp_utils
        import sys
        import os
        # 获取 utils 目录路径（与 executor/gp_executor.py 的方式一致）
        utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        from gp_utils.ridge_formula import generate_ridge_formula
        
        # 如果未启用High函数，不需要生成Ridge公式
        if not enable_high:
            return None
        
        # 检查必要参数
        if toolbox is None or train_features is None or train_labels is None:
            return None
        
        func = toolbox.compile(expr=individual)
        
        # 生成高阶特征（与 fitness.py 中的处理方式保持一致）
        train_features_high = []
        for i in range(len(train_labels)):
            try:
                pred_number = func(*train_features[i, :])
                # 检查NaN或Inf（与fitness.py保持一致）
                if np.any(np.isnan(pred_number)) or np.any(np.isinf(pred_number)):
                    return None
                train_features_high.append(pred_number)
            except (ZeroDivisionError, OverflowError):
                return None
            except TypeError as e:
                # 捕获类型错误（例如：High函数嵌套使用）
                if "root_con函数只接受Float1" in str(e) or "Vector1" in str(e):
                    return None
                # 其他TypeError也返回None
                return None
        
        train_high_features = np.array(train_features_high)
        
        # 确保特征是2D数组
        if train_high_features.ndim == 1:
            train_high_features = train_high_features.reshape(-1, 1)
        
        # 训练RidgeCV模型
        model = RidgeCV()
        model.fit(train_high_features, train_labels)
        
        # 生成公式
        gp_expression = str(individual)
        ridge_formula = generate_ridge_formula(model, gp_expression, target_variable)
        
        return ridge_formula
    except Exception:
        # 静默失败，返回None（避免影响主流程）
        return None

