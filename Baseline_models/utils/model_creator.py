"""
Baseline模型创建器
负责根据模型名称创建对应的模型实例
"""

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from DNN_models.dnn_model import DNNRegressor

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def create_baseline_model(model_name: str, input_size: int, random_seed: int = 1):
    """
    根据模型名称创建对应的模型实例
    
    参数:
        model_name: 模型名称
        input_size: 输入特征数量（某些模型可能需要）
        random_seed: 随机种子
    
    返回:
        模型实例
    """
    models = {
        # Linear Models
        'LinearRegression': LinearRegression(),
        'RidgeCV': RidgeCV(),
        'ElasticNet': ElasticNet(random_state=random_seed),
        
        # Tree Models
        'DecisionTree': DecisionTreeRegressor(random_state=random_seed),
        'DecisionTree_2': DecisionTreeRegressor(max_depth=2, random_state=random_seed),  # 限制深度为2的决策树
        'DecisionTree_4': DecisionTreeRegressor(max_depth=4, random_state=random_seed),  # 限制深度为4的决策树
        'DecisionTree_6': DecisionTreeRegressor(max_depth=6, random_state=random_seed),  # 限制深度为6的决策树
        'RandomForest': RandomForestRegressor(random_state=random_seed),
        'ExtraTrees': ExtraTreesRegressor(random_state=random_seed),
        'AdaBoost': AdaBoostRegressor(random_state=random_seed),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_seed),
        
        # Gradient Boosting SOTA Models
        'XGBoost': xgb.XGBRegressor(random_state=random_seed),
        'LightGBM': lgb.LGBMRegressor(random_state=random_seed, verbose=-1),
        
        # Neural Networks
        'MLP': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32, 16, 8),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.0005,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=random_seed
        ),
        
        # Other Methods
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'KNeighbors': KNeighborsRegressor(),

        # Deep Learning
        'DNN': DNNRegressor(
            hidden_layers=(512, 443, 374, 306, 237, 169, 100, 32),
            dropout=0.2,
            use_batch_norm=True,
            batch_size=128,
            learning_rate=0.001,
            max_epochs=2000,
            early_stopping_patience=30,
            validation_fraction=0.15,
            weight_decay=1e-5,
            device_preference="auto",
            random_seed=random_seed,
        ),
    }
    
    # 可选加入CatBoost
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostRegressor(
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
            train_dir=None,
            save_snapshot=False,
            snapshot_file=None,
            use_best_model=False
        )
    
    if model_name not in models:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    return models[model_name]

