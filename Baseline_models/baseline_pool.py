"""
Baseline模型池
负责管理可用的baseline模型配置
"""

import json
from typing import List


class Baseline_Pool:
    """
    Baseline模型池类
    
    参数:
        model_names: 用户希望注册激活的模型名称列表
    """
    
    # 所有可用的模型名称
    AVAILABLE_MODELS = [
        'LinearRegression',
        'RidgeCV',
        'ElasticNet',
        'DecisionTree',
        'RandomForest',
        'ExtraTrees',
        'AdaBoost',
        'GradientBoosting',
        'XGBoost',
        'LightGBM',
        'MLP',
        'SVR',
        'KNeighbors',
        'CatBoost',
        'DNN'
    ]
    
    def __init__(self, model_names: List[str]):
        """
        初始化Baseline模型池
        
        参数:
            model_names: 用户希望注册激活的模型名称列表
        """
        self.model_names = model_names if isinstance(model_names, list) else [model_names]
        
        # 验证模型名称是否有效
        invalid_models = [name for name in self.model_names if name not in self.AVAILABLE_MODELS]
        if invalid_models:
            raise ValueError(f"无效的模型名称: {invalid_models}。可用模型: {self.AVAILABLE_MODELS}")
    
    def get_config_json(self) -> dict:
        """
        获取模型配置JSON
        
        返回:
            dict: 包含模型配置的字典
        """
        config = {
            "available_models": self.AVAILABLE_MODELS,
            "selected_models": self.model_names,
            "model_count": len(self.model_names)
        }
        
        return config
    
    def get_config_json_str(self) -> str:
        """
        获取模型配置JSON字符串
        
        返回:
            str: JSON格式的配置字符串
        """
        return json.dumps(self.get_config_json(), indent=2, ensure_ascii=False)

