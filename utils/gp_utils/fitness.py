"""
GP适应度函数模块
包含用于进化过程中的适应度评估函数
- High函数模式：使用RidgeCV进行交叉验证
- 传统GP模式：直接使用R²评分
"""

import warnings
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 忽略numpy和sklearn的运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
np.seterr(all='ignore')  # 忽略所有numpy的浮点错误

# 无效个体的适应度值（大的负值）
INVALID_FITNESS_R2 = -1e8

# 无效误差指标的最大值
INVALID_ERROR_METRIC = 1e8


def which_fitness_to_use(individual, toolbox, train_features, train_labels, 
                         cv_folds=5, scale_factor=100, random_state=42):
    """
    根据个体类型自动选择适应的适应度函数
    如果个体是Vector类型（High函数模式），使用fitness_high_gp
    如果个体是Float类型（传统GP模式），使用eval_traditional_gp
    
    参数:
        individual: GP个体
        toolbox: DEAP工具箱
        train_features: 训练特征
        train_labels: 训练标签
        cv_folds: 交叉验证折数
        scale_factor: 适应度缩放因子（仅用于High函数模式）
        random_state: 随机种子，用于确保交叉验证划分的一致性
    
    返回:
        tuple: 适应度值
    """

    root_node = individual[0]
    is_high = (hasattr(root_node, 'name') and 
                root_node.name and 
                root_node.name.startswith('High_'))
    
    if is_high:
        # High函数模式：使用RidgeCV交叉验证
        return fitness_high_gp(
            individual, toolbox, train_features, train_labels,
            cv_folds=cv_folds, scale_factor=scale_factor, random_state=random_state
        )
    else:
        # 传统GP模式：使用交叉验证计算R²
        return eval_traditional_gp(
            individual, toolbox, train_features, train_labels,
            cv_folds=cv_folds, random_state=random_state
        )
        

def fitness_high_gp(individual, toolbox, train_features, train_labels, cv_folds=5, scale_factor=100, random_state=42):
    """
    High函数模式适应度函数
    使用RidgeCV进行交叉验证，将GP树输出作为特征进行Ridge回归
    
    输入：
        individual: GP个体
        toolbox: DEAP工具箱
        train_features: 训练特征 (n_samples, n_features)
        train_labels: 训练标签 (n_samples,)
        cv_folds: 交叉验证折数，默认5
        scale_factor: 适应度缩放因子，默认100
        random_state: 随机种子，用于确保交叉验证划分的一致性，默认42
    
    输出：
        tuple: (fitness_value,) - R²分数 * scale_factor，CV指标存储在individual.cv_metrics中
    """
    func = toolbox.compile(expr=individual)
    
    # 生成高阶特征
    train_features_high = []
    for i in range(len(train_labels)):
        try:
            pred_number = func(*train_features[i, :])
            # 检查NaN或Inf
            if np.any(np.isnan(pred_number)) or np.any(np.isinf(pred_number)):
                individual.cv_metrics = {}
                return (INVALID_FITNESS_R2,)  # 无效个体，返回大的负适应度
            train_features_high.append(pred_number)
        except (ZeroDivisionError, OverflowError):
            individual.cv_metrics = {}
            return (INVALID_FITNESS_R2,)  # 无效个体，返回大的负适应度
        except TypeError as e:
            # 捕获类型错误（例如：High函数嵌套使用）
            if "root_con函数只接受Float1" in str(e) or "Vector1" in str(e):
                individual.cv_metrics = {}
                return (INVALID_FITNESS_R2,)  # 无效个体（类型不匹配），返回大的负适应度
            else:
                # 其他TypeError，也返回大的负适应度
                individual.cv_metrics = {}
                return (INVALID_FITNESS_R2,)
    
    train_high_features = np.array(train_features_high)
    
    # 使用交叉验证评估，计算四个指标
    try:
        model = RidgeCV()
        
        # 创建带随机种子的KFold交叉验证对象，确保每次划分一致
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # 使用cross_validate获取多个指标
        cv_results = cross_validate(
            model, 
            train_high_features, 
            train_labels, 
            cv=kfold,
            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=False
        )
        
        # 计算R²均值作为适应度
        r2_scores = cv_results['test_r2']
        r2_mean = r2_scores.mean()
        
        if np.any(np.isnan(r2_scores)) or np.any(np.isinf(r2_scores)):
            individual.cv_metrics = {}
            return (INVALID_FITNESS_R2,)  # 如果出现NaN或Inf，返回大的负适应度
        
        # 计算四个指标的均值和标准差
        mse_scores = -cv_results['test_neg_mean_squared_error']  # 转换为正值
        mae_scores = -cv_results['test_neg_mean_absolute_error']  # 转换为正值
        rmse_scores = np.sqrt(mse_scores)
        
        # 构建新的cv_metrics格式：{folds: integer, cross_validation: {...}}
        cv_metrics = {
            'folds': cv_folds,
            'cross_validation': {
                'r2_mean': float(r2_mean),
                'r2_std': float(r2_scores.std()),
                'mse_mean': float(mse_scores.mean()),
                'mse_std': float(mse_scores.std()),
                'rmse_mean': float(rmse_scores.mean()),
                'rmse_std': float(rmse_scores.std()),
                'mae_mean': float(mae_scores.mean()),
                'mae_std': float(mae_scores.std())
            }
        }
        
        # 将CV指标存储在个体中
        individual.cv_metrics = cv_metrics

        fitness_value = round(scale_factor * r2_mean, 6)
        # 保持负适应度值，不强制设为0
        return (fitness_value,)
        
    except Exception:
        individual.cv_metrics = {}
        return (INVALID_FITNESS_R2,)  # 如果出现任何错误，返回大的负适应度


def eval_traditional_gp(individual, toolbox, X_train, y_train, cv_folds=5, random_state=42):
    """
    传统GP适应度评估函数
    不使用High函数，GP树直接返回标量值，使用交叉验证计算R²评分
    
    参数:
        individual: GP个体
        toolbox: 工具箱对象
        X_train: 训练特征
        y_train: 训练标签
        cv_folds: 交叉验证折数，默认5
        random_state: 随机种子，用于确保交叉验证划分的一致性，默认42
    
    返回:
        tuple: (r2_score_mean,) 交叉验证R²均值作为适应度值
        同时将CV指标保存到individual.cv_metrics中
    """
    try:
        # 编译个体为可执行函数
        func = toolbox.compile(expr=individual)
        
        # 创建带随机种子的KFold交叉验证对象，确保每次划分一致
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # 存储每一折的评估指标
        r2_scores = []
        mse_scores = []
        rmse_scores = []
        mae_scores = []
        
        # 对每一折进行交叉验证
        for train_idx, val_idx in kfold.split(X_train):
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # 计算验证集预测值
            y_pred_fold = []
            for i in range(len(y_val_fold)):
                try:
                    y_pred_number = func(*X_val_fold[i, :])
                    # 检查是否为有效数值
                    if not np.isfinite(y_pred_number):
                        y_pred_number = 0.0
                    y_pred_fold.append(y_pred_number)
                except:
                    y_pred_fold.append(0.0)
            
            # 转换为numpy数组并检查NaN
            y_pred_fold = np.array(y_pred_fold)
            y_pred_fold = np.nan_to_num(y_pred_fold, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # 计算该折的评估指标
            try:
                r2_fold = r2_score(y_val_fold, y_pred_fold)
                mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
                rmse_fold = np.sqrt(mse_fold)
                mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
                
                # 如果R²为NaN或无效，使用无效值
                if not np.isfinite(r2_fold):
                    r2_fold = INVALID_FITNESS_R2
                    mse_fold = INVALID_ERROR_METRIC
                    rmse_fold = INVALID_ERROR_METRIC
                    mae_fold = INVALID_ERROR_METRIC
            except:
                # 如果计算失败，使用无效值
                r2_fold = INVALID_FITNESS_R2
                mse_fold = INVALID_ERROR_METRIC
                rmse_fold = INVALID_ERROR_METRIC
                mae_fold = INVALID_ERROR_METRIC
            
            r2_scores.append(r2_fold)
            mse_scores.append(mse_fold)
            rmse_scores.append(rmse_fold)
            mae_scores.append(mae_fold)
        
        # 转换为numpy数组
        r2_scores = np.array(r2_scores)
        mse_scores = np.array(mse_scores)
        rmse_scores = np.array(rmse_scores)
        mae_scores = np.array(mae_scores)
        
        # 计算均值
        r2_mean = r2_scores.mean()
        
        # 如果所有折的R²都无效，返回大的负适应度
        if np.any(np.isnan(r2_scores)) or np.any(np.isinf(r2_scores)) or not np.isfinite(r2_mean):
            individual.cv_metrics = {}
            return (INVALID_FITNESS_R2,)
        
        # 构建cv_metrics格式：{folds: integer, cross_validation: {...}}
        cv_metrics = {
            'folds': cv_folds,
            'cross_validation': {
                'r2_mean': float(r2_mean),
                'r2_std': float(r2_scores.std()),
                'mse_mean': float(mse_scores.mean()),
                'mse_std': float(mse_scores.std()),
                'rmse_mean': float(rmse_scores.mean()),
                'rmse_std': float(rmse_scores.std()),
                'mae_mean': float(mae_scores.mean()),
                'mae_std': float(mae_scores.std())
            }
        }
        
        # 将CV指标存储在个体中
        individual.cv_metrics = cv_metrics
        
        return (r2_mean,)
        
    except Exception as e:
        # 如果评估失败，返回大的负适应度
        individual.cv_metrics = {}
        return (INVALID_FITNESS_R2,)

