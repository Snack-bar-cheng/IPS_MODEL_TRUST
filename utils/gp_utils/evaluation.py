"""
GP评估函数模块
包含测试集评估和残差拟合评估函数
"""

import os
import sys
import time
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 尝试加载 DNN 残差模型（与 Baseline 目录中的实现保持一致）
BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Baseline_models'))
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

# 引入utils目录，加载SHAP工具
UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if UTILS_ROOT not in sys.path:
    sys.path.insert(0, UTILS_ROOT)

try:
    from DNN_models.dnn_model import DNNRegressor
    DNN_AVAILABLE = True
except ImportError:
    DNN_AVAILABLE = False

try:
    from feature_importance.shap_handler import compute_shap_importances
    from feature_importance.shap_sampler import sample_with_ks
except Exception:
    compute_shap_importances = None
    sample_with_ks = None


def evalSymbReg_Test(individual, toolbox, train_features, train_labels, test_features, test_labels, enable_high=True):
    """
    测试集适应度评估函数
    支持High函数模式和传统GP模式
    
    输入：
        individual: GP个体
        toolbox: DEAP工具箱
        train_features: 训练特征 (n_samples, n_features)
        train_labels: 训练标签 (n_samples,)
        test_features: 测试特征 (n_test_samples, n_features)
        test_labels: 测试标签 (n_test_samples,)
        enable_high: 是否启用High函数模式，默认True
    
    输出：
        tuple: (test_metrics, train_metrics, model, gp_expression)
            - test_metrics: 测试集指标字典 {'r2': float, 'mse': float, 'rmse': float, 'mae': float}
            - train_metrics: 训练集指标字典 {'r2': float, 'mse': float, 'rmse': float, 'mae': float}
            - model: 训练好的模型（High函数模式为RidgeCV，传统GP为None）
            - gp_expression: GP表达式字符串
    """
    func = toolbox.compile(expr=individual)
    
    if enable_high:
        # High函数模式：GP生成高阶特征，然后用RidgeCV拟合
        # 生成训练集高阶特征
        train_features_high = []
        for i in range(len(train_labels)):
            pred_number = func(*train_features[i, :])
            train_features_high.append(pred_number)
        
        train_high_features = np.array(train_features_high)
        
        # 生成测试集高阶特征
        test_features_high = []
        for i in range(len(test_labels)):
            pred_number = func(*test_features[i, :])
            test_features_high.append(pred_number)
        
        test_high_features = np.array(test_features_high)
        
        # 确保特征是2D数组 (如果是1D，reshape为2D)
        if train_high_features.ndim == 1:
            train_high_features = train_high_features.reshape(-1, 1)
        if test_high_features.ndim == 1:
            test_high_features = test_high_features.reshape(-1, 1)
        
        # 训练模型并预测
        model = RidgeCV()
        model.fit(train_high_features, train_labels)
        y_pred_train = model.predict(train_high_features)
        y_pred_test = model.predict(test_high_features)
    else:
        # 传统GP模式：GP直接预测标量值
        # 计算训练集预测值
        y_pred_train = []
        for i in range(len(train_labels)):
            try:
                y_pred_number = func(*train_features[i, :])
                if not np.isfinite(y_pred_number):
                    y_pred_number = 0.0
                y_pred_train.append(y_pred_number)
            except:
                y_pred_train.append(0.0)
        y_pred_train = np.array(y_pred_train)
        y_pred_train = np.nan_to_num(y_pred_train, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # 计算测试集预测值
        y_pred_test = []
        for i in range(len(test_labels)):
            try:
                y_pred_number = func(*test_features[i, :])
                if not np.isfinite(y_pred_number):
                    y_pred_number = 0.0
                y_pred_test.append(y_pred_number)
            except:
                y_pred_test.append(0.0)
        y_pred_test = np.array(y_pred_test)
        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=1e10, neginf=-1e10)
        
        model = None  # 传统GP没有模型
    
    # 计算评估指标
    train_r2 = r2_score(train_labels, y_pred_train)
    train_mse = mean_squared_error(train_labels, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_labels, y_pred_train)
    
    test_r2 = r2_score(test_labels, y_pred_test)
    test_mse = mean_squared_error(test_labels, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_labels, y_pred_test)
    
    return {
        'r2': test_r2,
        'mse': test_mse,
        'rmse': test_rmse,
        'mae': test_mae
    }, {
        'r2': train_r2,
        'mse': train_mse,
        'rmse': train_rmse,
        'mae': train_mae
    }, model, str(individual)


def _create_residual_model(model_name, random_seed=42):
    """
    创建残差拟合模型
    
    输入：
        model_name: 模型名称，支持 'CatBoost', 'ExtraTrees', 'RandomForest', 'GradientBoosting', 'LightGBM', 'RidgeCV', 'DNN'
        random_seed: 随机种子，默认42
    
    输出：
        模型实例
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'catboost':
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=100, random_seed=random_seed, verbose=False, allow_writing_files=False)
        except ImportError:
            raise ImportError(f"CatBoost not installed. Please install it with: pip install catboost")
    
    elif model_name_lower == 'extratrees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
    
    elif model_name_lower == 'randomforest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
    
    elif model_name_lower == 'gradientboosting':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, random_state=random_seed)
    
    elif model_name_lower == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=random_seed, verbose=-1, n_jobs=-1)
        except ImportError:
            raise ImportError(f"LightGBM not installed. Please install it with: pip install lightgbm")
    
    elif model_name_lower == 'ridgecv':
        return RidgeCV()

    elif model_name_lower == 'dnn':
        if not DNN_AVAILABLE:
            raise ImportError("DNN module not found. Please ensure Baseline_models/DNN_models/dnn_model.py is importable.")
        return DNNRegressor(random_seed=random_seed)
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: CatBoost, ExtraTrees, RandomForest, GradientBoosting, LightGBM, RidgeCV, DNN")


def evalSymbReg_Test_with_residual_fitting(
    individual,
    toolbox,
    train_features,
    train_labels,
    test_features,
    test_labels,
    residual_models=None,
    random_seed=42,
    enable_high=True,
    shap_open=False,
    shap_save_dir=None,
    feature_names=None,
    target_name="",
    shap_bg_size=None,
    shap_explain_size=None,
    shap_ks_threshold=0.05,
    shap_max_attempts=5,
):
    """
    测试集适应度评估函数（带残差拟合）
    支持High函数模式和传统GP模式
    支持多个残差拟合模型
    
    输入：
        individual: GP个体
        toolbox: DEAP工具箱
        train_features: 训练特征 (n_samples, n_features)
        train_labels: 训练标签 (n_samples,)
        test_features: 测试特征 (n_test_samples, n_features)
        test_labels: 测试标签 (n_test_samples,)
        residual_models: 残差拟合模型列表，例如 ['CatBoost', 'ExtraTrees', 'RandomForest', 'GradientBoosting', 'LightGBM']
                        如果为None或空列表，默认使用 ['RandomForest']
        random_seed: 随机种子，默认42
        enable_high: 是否启用High函数模式，默认True。True时使用RidgeCV拟合GP生成的高阶特征，False时GP直接预测（传统GP模式）
    
    输出：
        dict: 包含原始结果和残差拟合后结果的字典
        {
            'original': {
                'train': {'r2': float, 'mse': float, 'rmse': float, 'mae': float},
                'test': {'r2': float, 'mse': float, 'rmse': float, 'mae': float},
                'model': trained_model,
                'gp_expression': str
            },
            'with_residual': [
                {
                    'model_name': str,  # 模型名称
                    'train': {'r2': float, 'mse': float, 'rmse': float, 'mae': float},
                    'test': {'r2': float, 'mse': float, 'rmse': float, 'mae': float},
                    'residual_model': trained_residual_model,
                    'residual_contribution': dict,  # 残差贡献信息（特征重要性等）
                    'residual_training_duration': float  # 残差模型训练时间（秒）
                },
                ...
            ]
        }
    """
    # 默认使用 RandomForest 如果未指定模型
    if residual_models is None or len(residual_models) == 0:
        residual_models = ['RandomForest']
    
    # ========== GP生成预测值或高阶特征的过程 ==========
    func = toolbox.compile(expr=individual)
    
    if enable_high:
        # High函数模式：GP生成高阶特征，然后用RidgeCV拟合
        # 生成训练集高阶特征
        train_features_high = []
        for i in range(len(train_labels)):
            try:
                pred_number = func(*train_features[i, :])
                if np.any(np.isnan(pred_number)) or np.any(np.isinf(pred_number)):
                    return {
                        'original': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'gp_expression': str(individual)},
                        'with_residual': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'residual_model': None, 'residual_contribution': {}}
                    }
                train_features_high.append(pred_number)
            except (ZeroDivisionError, OverflowError, TypeError):
                return {
                    'original': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'gp_expression': str(individual)},
                    'with_residual': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'residual_model': None, 'residual_contribution': {}}
                }
        
        train_high_features = np.array(train_features_high)
        
        # 生成测试集高阶特征
        test_features_high = []
        for i in range(len(test_labels)):
            try:
                pred_number = func(*test_features[i, :])
                if np.any(np.isnan(pred_number)) or np.any(np.isinf(pred_number)):
                    return {
                        'original': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'gp_expression': str(individual)},
                        'with_residual': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'residual_model': None, 'residual_contribution': {}}
                    }
                test_features_high.append(pred_number)
            except (ZeroDivisionError, OverflowError, TypeError):
                return {
                    'original': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'gp_expression': str(individual)},
                    'with_residual': {'train': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'test': {'r2': 0, 'mse': np.inf, 'rmse': np.inf, 'mae': np.inf}, 'model': None, 'residual_model': None, 'residual_contribution': {}}
                }
        
        test_high_features = np.array(test_features_high)
        
        # 确保特征是2D数组 (如果是1D，reshape为2D)
        if train_high_features.ndim == 1:
            train_high_features = train_high_features.reshape(-1, 1)
        if test_high_features.ndim == 1:
            test_high_features = test_high_features.reshape(-1, 1)
        
        # ========== 原始模型（RidgeCV） ==========
        original_model = RidgeCV()
        original_model.fit(train_high_features, train_labels)
        y_pred_train = original_model.predict(train_high_features)
        y_pred_test = original_model.predict(test_high_features)
    else:
        # 传统GP模式：GP直接预测标量值，不使用RidgeCV
        # 计算训练集预测值
        y_pred_train = []
        for i in range(len(train_labels)):
            try:
                y_pred_number = func(*train_features[i, :])
                if not np.isfinite(y_pred_number):
                    y_pred_number = 0.0
                y_pred_train.append(y_pred_number)
            except:
                y_pred_train.append(0.0)
        y_pred_train = np.array(y_pred_train)
        y_pred_train = np.nan_to_num(y_pred_train, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # 计算测试集预测值
        y_pred_test = []
        for i in range(len(test_labels)):
            try:
                y_pred_number = func(*test_features[i, :])
                if not np.isfinite(y_pred_number):
                    y_pred_number = 0.0
                y_pred_test.append(y_pred_number)
            except:
                y_pred_test.append(0.0)
        y_pred_test = np.array(y_pred_test)
        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=1e10, neginf=-1e10)
        
        original_model = None  # 传统GP没有模型
    
    # 计算原始模型在训练集和测试集上的评估指标
    original_train_r2 = r2_score(train_labels, y_pred_train)
    original_train_mse = mean_squared_error(train_labels, y_pred_train)
    original_train_rmse = np.sqrt(original_train_mse)
    original_train_mae = mean_absolute_error(train_labels, y_pred_train)
    
    original_test_r2 = r2_score(test_labels, y_pred_test)
    original_test_mse = mean_squared_error(test_labels, y_pred_test)
    original_test_rmse = np.sqrt(original_test_mse)
    original_test_mae = mean_absolute_error(test_labels, y_pred_test)
    
    # ========== 残差拟合 ==========
    # 计算训练集残差：真实值 - 原始模型预测值
    train_residuals = train_labels - y_pred_train
    
    # ========== 对每个模型进行残差拟合 ==========
    with_residual_list = []
    
    for model_name in residual_models:
        try:
            # 创建残差模型
            residual_model = _create_residual_model(model_name, random_seed=random_seed)
            
            # ========== 残差模型训练时间计算 ==========
            residual_training_start_time = time.time()
            
            # 使用原始特征拟合残差
            residual_model.fit(train_features, train_residuals)
            residual_pred_train = residual_model.predict(train_features)
            residual_pred_test = residual_model.predict(test_features)
            
            # ========== 残差模型训练时间计算结束 ==========
            residual_training_end_time = time.time()
            residual_training_duration = residual_training_end_time - residual_training_start_time
            
            # 合并预测结果（训练集和测试集）
            final_pred_train = y_pred_train + residual_pred_train
            final_pred_test = y_pred_test + residual_pred_test
            
            # 计算残差拟合后在训练集和测试集上的评估指标
            residual_train_r2 = r2_score(train_labels, final_pred_train)
            residual_train_mse = mean_squared_error(train_labels, final_pred_train)
            residual_train_rmse = np.sqrt(residual_train_mse)
            residual_train_mae = mean_absolute_error(train_labels, final_pred_train)
            
            residual_test_r2 = r2_score(test_labels, final_pred_test)
            residual_test_mse = mean_squared_error(test_labels, final_pred_test)
            residual_test_rmse = np.sqrt(residual_test_mse)
            residual_test_mae = mean_absolute_error(test_labels, final_pred_test)
            
            # ========== 计算IPS (Interpretable Performance Share) ==========
            # 计算基线（均值预测）的MSE和R²
            y_mean_train = np.mean(train_labels)
            y_mean_test = np.mean(test_labels)
            
            baseline_train_mse = mean_squared_error(train_labels, np.full_like(train_labels, y_mean_train))
            baseline_test_mse = mean_squared_error(test_labels, np.full_like(test_labels, y_mean_test))
            baseline_train_r2 = r2_score(train_labels, np.full_like(train_labels, y_mean_train))
            baseline_test_r2 = r2_score(test_labels, np.full_like(test_labels, y_mean_test))
            
            # 计算样本数
            n_train = len(train_labels)
            n_test = len(test_labels)
            
            # 计算SSE（从MSE转换）
            SSE_0_train = baseline_train_mse * n_train
            SSE_0_test = baseline_test_mse * n_test
            SSE_g_train = original_train_mse * n_train
            SSE_g_test = original_test_mse * n_test
            SSE_g_h_train = residual_train_mse * n_train
            SSE_g_h_test = residual_test_mse * n_test
            
            # 计算IPS_R2: R²_g / R²_{g+h}
            def calculate_ips_r2(r2_g, r2_total):
                """计算IPS_R2"""
                ips = r2_g / r2_total
                return float(ips)
            
            # 计算IPS_SSE: (SSE_0 - SSE_g) / (SSE_0 - SSE_{g+h})
            def calculate_ips_sse(sse_0, sse_g, sse_total):
                """计算IPS_SSE"""
                denominator = sse_0 - sse_total
                ips = (sse_0 - sse_g) / denominator
                return float(ips)
            
            # 计算训练集的IPS
            ips_r2_train = calculate_ips_r2(original_train_r2, residual_train_r2)
            ips_sse_train = calculate_ips_sse(SSE_0_train, SSE_g_train, SSE_g_h_train)
            
            # 计算测试集的IPS
            ips_r2_test = calculate_ips_r2(original_test_r2, residual_test_r2)
            ips_sse_test = calculate_ips_sse(SSE_0_test, SSE_g_test, SSE_g_h_test)
            
            # ========== 残差贡献分析 ==========
            residual_contribution = {}
            shap_plot_path = None
            shap_bee_path = None
            interaction_importances = None

            # 优先尝试SHAP（树模型）
            if shap_open and compute_shap_importances is not None:
                print(f"[SHAP] 正在计算残差模型的SHAP值... 模型={model_name}, 目标={target_name}")
                X_bg = train_features
                X_explain = test_features
                if sample_with_ks is not None:
                    X_bg, X_explain = sample_with_ks(
                        train_features, test_features,
                        background_size=shap_bg_size,
                        explain_size=shap_explain_size,
                        ks_threshold=shap_ks_threshold,
                        max_attempts=shap_max_attempts,
                        random_state=random_seed
                    )
                
                # 输出背景集和解释集的数据条数
                print(f"[SHAP] 背景集数据条数: {X_bg.shape[0]} (配置: {shap_bg_size if shap_bg_size is not None else 'None-使用全部训练集'})")
                print(f"[SHAP] 解释集数据条数: {X_explain.shape[0]} (配置: {shap_explain_size if shap_explain_size is not None else 'None-使用全部测试集'})")

                fi_payload, inter_payload, shap_plot_path, shap_bee_path = compute_shap_importances(
                    model=residual_model,
                    X_train=X_bg,
                    X_explain=X_explain,
                    feature_names=feature_names,
                    target_name=target_name,
                    model_name=model_name,
                    random_seed=random_seed,
                    save_dir=shap_save_dir,
                    prefix="residual",
                    background_limit=None,  # None表示使用全部采样后的背景数据
                    explain_limit=None,     # None表示使用全部采样后的解释数据
                )
                if fi_payload is not None:
                    residual_contribution['feature_importances'] = fi_payload
                if inter_payload is not None:
                    interaction_importances = inter_payload
                    residual_contribution['interaction_feature_importances'] = inter_payload
                if shap_plot_path:
                    residual_contribution['shap_plot_path'] = shap_plot_path
                if shap_bee_path:
                    residual_contribution['shap_beeswarm_plot_path'] = shap_bee_path

            # 计算残差预测的统计信息
            residual_std = np.std(train_residuals)
            residual_mean = np.mean(np.abs(train_residuals))
            residual_train_r2_model = r2_score(train_residuals, residual_pred_train)
            residual_test_r2_model = r2_score(test_labels - y_pred_test, residual_pred_test) if len(test_labels) > 0 else 0

            residual_contribution.update({
                'residual_std': float(residual_std),
                'residual_mean_abs': float(residual_mean),
                'residual_train_r2': float(residual_train_r2_model),
                'residual_test_r2': float(residual_test_r2_model),
                'residual_pred_train_mean': float(np.mean(np.abs(residual_pred_train))),
                'residual_pred_test_mean': float(np.mean(np.abs(residual_pred_test))),
                'residual_pred_train_std': float(np.std(residual_pred_train)),
                'residual_pred_test_std': float(np.std(residual_pred_test))
            })

            # 回退到传统feature_importances_或系数
            if 'feature_importances' not in residual_contribution and hasattr(residual_model, 'feature_importances_'):
                feature_importances = residual_model.feature_importances_
                residual_contribution['feature_importances'] = feature_importances.tolist() if hasattr(feature_importances, 'tolist') else list(feature_importances)
            elif 'feature_importances' not in residual_contribution and hasattr(residual_model, 'coef_'):
                coef = residual_model.coef_
                residual_contribution.update({
                    'coefficients': coef.tolist() if hasattr(coef, 'tolist') else list(coef),
                    'intercept': float(residual_model.intercept_) if hasattr(residual_model, 'intercept_') else 0.0
                })

            # 添加到结果列表
            train_metrics_dict = {
                'r2': residual_train_r2,
                'mse': residual_train_mse,
                'rmse': residual_train_rmse,
                'mae': residual_train_mae,
                'ips_r2': ips_r2_train,
                'ips_sse': ips_sse_train
            }
            
            test_metrics_dict = {
                'r2': residual_test_r2,
                'mse': residual_test_mse,
                'rmse': residual_test_rmse,
                'mae': residual_test_mae,
                'ips_r2': ips_r2_test,
                'ips_sse': ips_sse_test
            }
            
            with_residual_list.append({
                'model_name': model_name,
                'train': train_metrics_dict,
                'test': test_metrics_dict,
                'residual_model': residual_model,  # 残差模型
                'residual_contribution': residual_contribution,  # 残差贡献信息
                'residual_training_duration': residual_training_duration  # 残差模型训练时间（秒）
            })
            
        except Exception as e:
            # 如果某个模型训练失败，记录错误也写入结果列表，避免为空
            print(f"Warning: Failed to train residual model {model_name}: {e}")
            with_residual_list.append({
                'model_name': model_name,
                'error': str(e),
                'train': {},
                'test': {},
                'residual_contribution': {'error': str(e)}
            })
            continue
    
    return {
        'original': {
            'train': {
                'r2': original_train_r2,
                'mse': original_train_mse,
                'rmse': original_train_rmse,
                'mae': original_train_mae
            },
            'test': {
                'r2': original_test_r2,
                'mse': original_test_mse,
                'rmse': original_test_rmse,
                'mae': original_test_mae
            },
            'model': original_model,
            'gp_expression': str(individual)
        },
        'with_residual': with_residual_list
    }
