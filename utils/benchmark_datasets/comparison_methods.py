"""
对比方法实现
用于与你的混合模型方法对比
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def polynomial_features_ridge(X_train, y_train, X_test, y_test, 
                               degree: int = 2, interaction_only: bool = False) -> Dict:
    """
    Polynomial Features + RidgeCV（核心对比方法）
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        degree: 多项式次数
        interaction_only: 是否只包含交互项
    
    返回:
        dict: 包含预测结果和评估指标
    """
    # 生成多项式特征
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 训练RidgeCV模型
    model = RidgeCV()
    model.fit(X_train_poly, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    # 计算指标
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    return {
        'method': 'PolynomialFeatures_Ridge',
        'train': {
            'r2': train_r2,
            'rmse': train_rmse,
            'mae': train_mae
        },
        'test': {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        },
        'n_features': X_train_poly.shape[1],
        'model': model,
        'poly_transformer': poly
    }


def random_forest_baseline(X_train, y_train, X_test, y_test, 
                          n_estimators: int = 100) -> Dict:
    """
    RandomForest（原始特征）- 基准对比
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        n_estimators: 树的数量
    
    返回:
        dict: 包含预测结果和评估指标
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    return {
        'method': 'RandomForest',
        'train': {
            'r2': train_r2,
            'rmse': train_rmse,
            'mae': train_mae
        },
        'test': {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        },
        'n_features': X_train.shape[1],
        'model': model,
        'feature_importances': model.feature_importances_
    }


def xgboost_baseline(X_train, y_train, X_test, y_test) -> Dict:
    """
    XGBoost（原始特征）- 基准对比
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
    
    返回:
        dict: 包含预测结果和评估指标
    """
    try:
        import xgboost as xgb
        
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        return {
            'method': 'XGBoost',
            'train': {
                'r2': train_r2,
                'rmse': train_rmse,
                'mae': train_mae
            },
            'test': {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            },
            'n_features': X_train.shape[1],
            'model': model,
            'feature_importances': model.feature_importances_
        }
    except ImportError:
        logger.warning("XGBoost not installed, skipping XGBoost baseline")
        return None


def neural_network_baseline(X_train, y_train, X_test, y_test,
                            hidden_layer_sizes: Tuple = (100, 50)) -> Dict:
    """
    Neural Network（原始特征）- 基准对比
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        hidden_layer_sizes: 隐藏层大小
    
    返回:
        dict: 包含预测结果和评估指标
    """
    try:
        from sklearn.neural_network import MLPRegressor
        
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        return {
            'method': 'NeuralNetwork',
            'train': {
                'r2': train_r2,
                'rmse': train_rmse,
                'mae': train_mae
            },
            'test': {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            },
            'n_features': X_train.shape[1],
            'model': model
        }
    except Exception as e:
        logger.warning(f"Neural Network failed: {e}")
        return None


def run_all_baselines(X_train, y_train, X_test, y_test) -> Dict:
    """
    运行所有基准对比方法
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
    
    返回:
        dict: 包含所有方法的实验结果
    """
    results = {}
    
    # Polynomial Features + Ridge（核心对比）
    logger.info("Running Polynomial Features + Ridge...")
    results['poly_ridge_deg2'] = polynomial_features_ridge(X_train, y_train, X_test, y_test, degree=2)
    results['poly_ridge_deg3'] = polynomial_features_ridge(X_train, y_train, X_test, y_test, degree=3)
    
    # RandomForest
    logger.info("Running RandomForest...")
    results['random_forest'] = random_forest_baseline(X_train, y_train, X_test, y_test)
    
    # XGBoost
    logger.info("Running XGBoost...")
    xgb_result = xgboost_baseline(X_train, y_train, X_test, y_test)
    if xgb_result:
        results['xgboost'] = xgb_result
    
    # Neural Network
    logger.info("Running Neural Network...")
    nn_result = neural_network_baseline(X_train, y_train, X_test, y_test)
    if nn_result:
        results['neural_network'] = nn_result
    
    return results

