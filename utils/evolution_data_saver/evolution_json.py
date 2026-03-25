"""
进化JSON更新模块
负责更新进化过程JSON文件
"""

import json
import os
from typing import List, Optional, Dict

from .utils import format_json_compact, convert_to_relative_path


def update_evolution_json_with_ridge_formula(evolution_json_path, ridge_formula, train_metrics, test_metrics, best_expression, cv_metrics=None, gp_hyperparams=None, split_dataset_paths=None, residual_fitting_results=None, training_duration=None, residual_training_duration=None, baseline_info=None):
    """
    更新进化过程JSON文件，添加Ridge公式和最终指标（新结构）
    
    参数:
        evolution_json_path: 进化过程JSON文件路径
        ridge_formula: Ridge回归公式
        train_metrics: 训练集指标
        test_metrics: 测试集指标
        best_expression: 最佳个体的GP表达式
        cv_metrics: 五折交叉验证指标
        gp_hyperparams: GP超参数配置
        split_dataset_paths: 数据分割文件路径字典（已废弃，使用train_file_path和test_file_path）
        residual_fitting_results: 残差拟合结果字典，包含：
            - original: 原始模型指标（train和test）
            - with_residual: 残差拟合后指标（train和test）
            - residual_contribution: 残差贡献信息（特征重要性等）
        training_duration: 核心训练时间（秒）
        residual_training_duration: 残差模型训练时间（已废弃，不再使用。
            每个残差模型的训练时间已包含在 residual_fitting_results 的 with_residual 列表中）
        baseline_info: baseline评估结果字典，格式为 {"baseline_models": [...]}
    """
    try:
        if os.path.exists(evolution_json_path):
            with open(evolution_json_path, 'r', encoding='utf-8') as f:
                evolution_data = json.load(f)
            
            # 确保新结构存在
            if "gp_info" not in evolution_data:
                evolution_data["gp_info"] = {}
            
            # 添加GP超参数到gp_info中（如果提供了）
            if gp_hyperparams:
                gp_hyperparams_with_desc = gp_hyperparams.copy()
                gp_hyperparams_with_desc["description"] = "Genetic Programming hyperparameters used in this experiment"
                evolution_data["gp_info"]["gp_hyperparameters"] = gp_hyperparams_with_desc
            
            # 更新experiment_info中的数据分割路径（如果提供了）
            if split_dataset_paths and "experiment_info" in evolution_data:
                if "train_file" in split_dataset_paths:
                    evolution_data["experiment_info"]["train_file_path"] = split_dataset_paths["train_file"]
                if "test_file" in split_dataset_paths:
                    evolution_data["experiment_info"]["test_file_path"] = split_dataset_paths["test_file"]
            
            # 构建最终模型信息（放在gp_info下）
            final_model_info = {
                "gp_expression": best_expression,
                "ridge_formula": ridge_formula,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "note": "Train and test metrics show final model performance without cross-validation. But during evolution, 5-fold cross-validation is used to prevent overfitting."
            }
            
            # 添加训练时间信息到final_model_info（如果提供了）
            if training_duration is not None:
                final_model_info["training_time"] = {
                    "duration_seconds": float(training_duration),
                    "duration_minutes": float(training_duration / 60),
                    "duration_hours": float(training_duration / 3600),
                    "description": "核心训练时间：只包括进化过程（从初始种群评估到所有代完成），不包括数据加载、系统设置、结果保存等操作"
                }
            
            # 添加CV指标（如果存在），转换为新格式
            if cv_metrics:
                # 支持两种格式：
                # 1. 新格式：{folds: 5, cross_validation: {...}}
                # 2. 旧格式：{folds: [...], summary: {...}}
                if isinstance(cv_metrics, dict):
                    # 检查是否已经是新格式
                    if 'cross_validation' in cv_metrics:
                        # 已经是新格式，直接使用
                        final_model_info["cv_metrics"] = {
                            "folds": cv_metrics.get('folds', 5),
                            "cross_validation": cv_metrics.get('cross_validation', {})
                        }
                    elif 'summary' in cv_metrics:
                        # 旧格式，需要转换
                        summary = cv_metrics.get('summary', {})
                        folds_list = cv_metrics.get('folds', [])
                        folds_count = len(folds_list) if isinstance(folds_list, list) else 5
                        
                        # 构建新的cv_metrics格式
                        final_model_info["cv_metrics"] = {
                            "folds": folds_count,
                            "cross_validation": {
                                "r2_mean": summary.get('r2_mean', 0),
                                "r2_std": summary.get('r2_std', 0),
                                "mse_mean": summary.get('mse_mean', 0),
                                "mse_std": summary.get('mse_std', 0),
                                "rmse_mean": summary.get('rmse_mean', 0),
                                "rmse_std": summary.get('rmse_std', 0),
                                "mae_mean": summary.get('mae_mean', 0),
                                "mae_std": summary.get('mae_std', 0)
                            }
                        }
                    else:
                        # 格式不正确，使用默认值
                        final_model_info["cv_metrics"] = {
                            "folds": 5,
                            "cross_validation": {}
                        }
                else:
                    # 如果格式不正确，使用默认值
                    final_model_info["cv_metrics"] = {
                        "folds": 5,
                        "cross_validation": {}
                    }
            
            # 添加残差拟合结果（如果存在；允许空dict以保留结构）
            if residual_fitting_results is not None:
                # 获取残差模型列表（新格式）或单个模型（旧格式兼容）
                with_residual = residual_fitting_results.get('with_residual', [])
                
                # 兼容旧格式：如果是字典，转换为列表
                if isinstance(with_residual, dict):
                    # 旧格式：单个模型
                    residual_model_name = "RandomForest"  # 默认值
                    residual_model = with_residual.get('residual_model')
                    if residual_model is not None:
                        try:
                            class_name = residual_model.__class__.__name__
                            if 'RandomForest' in class_name:
                                residual_model_name = "RandomForest"
                            elif 'Ridge' in class_name:
                                residual_model_name = "RidgeCV"
                        except:
                            pass
                    
                    with_residual = [{
                        'model_name': residual_model_name,
                        'train': with_residual.get('train', {}),
                        'test': with_residual.get('test', {}),
                        'residual_contribution': with_residual.get('residual_contribution', {}),
                        'residual_training_duration': with_residual.get('residual_training_duration')
                    }]
                
                # 新格式：多个模型列表
                with_residual_fitting_list = []
                for residual_item in with_residual:
                    model_name = residual_item.get('model_name', 'Unknown')
                    train_metrics = residual_item.get('train', {})
                    test_metrics = residual_item.get('test', {})
                    residual_training_duration = residual_item.get('residual_training_duration')
                    residual_contribution = residual_item.get('residual_contribution', {})
                    
                    model_info = {
                        "modelname": model_name,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics
                    }
                    
                    # 添加训练时间（如果存在）
                    if residual_training_duration is not None:
                        model_info["training_duration"] = {
                            "seconds": float(residual_training_duration),
                            "minutes": float(residual_training_duration / 60),
                            "hours": float(residual_training_duration / 3600)
                        }
                    
                    # 添加残差特征重要性到当前模型项下（统一添加，不支持时设为null）
                    if residual_contribution and 'feature_importances' in residual_contribution:
                        feature_importances = residual_contribution.get('feature_importances', [])
                        feature_names = residual_contribution.get('feature_names', [])
                        
                        if isinstance(feature_importances, list) and feature_importances and isinstance(feature_importances[0], dict):
                            model_info["residual_feature_importances"] = feature_importances
                        else:
                            # 按重要性排序（旧格式）
                            sorted_importances = sorted(enumerate(feature_importances), key=lambda x: x[1], reverse=True)
                        residual_feature_importance_list = []
                        for feat_idx, importance in sorted_importances:
                            feature_item = {
                                "importance": float(importance)
                            }
                            # 如果有特征名称，也保存
                            if feature_names and feat_idx < len(feature_names):
                                feature_item["feature_name"] = feature_names[feat_idx]
                            residual_feature_importance_list.append(feature_item)
                        model_info["residual_feature_importances"] = residual_feature_importance_list
                    else:
                        model_info["residual_feature_importances"] = None

                    # 交互特征重要性（SHAP）
                    interaction_importances = residual_contribution.get("interaction_feature_importances")
                    if interaction_importances and isinstance(interaction_importances, list):
                        model_info["residual_interaction_feature_importances"] = interaction_importances
                    else:
                        model_info["residual_interaction_feature_importances"] = None

                    # SHAP图路径（若存在则转换为相对路径）
                    shap_plot_path = residual_contribution.get("shap_plot_path")
                    if shap_plot_path:
                        model_info["residual_shap_plot_path"] = convert_to_relative_path(
                            shap_plot_path, os.path.dirname(evolution_json_path)
                        )
                    shap_bee_path = residual_contribution.get("shap_beeswarm_plot_path")
                    if shap_bee_path:
                        model_info["residual_shap_beeswarm_plot_path"] = convert_to_relative_path(
                            shap_bee_path, os.path.dirname(evolution_json_path)
                        )
                    
                    with_residual_fitting_list.append(model_info)
                
                # 添加 with_residual_fitting 列表
                final_model_info["with_residual_fitting"] = with_residual_fitting_list
            
            # 更新最后一代的累计时间，确保它等于 training_duration
            # 这样可以保证累计时间的准确性和一致性
            if training_duration is not None and "gp_info" in evolution_data and "generations" in evolution_data["gp_info"]:
                generations = evolution_data["gp_info"]["generations"]
                if generations:
                    last_gen = generations[-1]
                    last_gen["cumulative_time_seconds"] = float(training_duration)
            
            evolution_data["gp_info"]["gp_final_model"] = final_model_info
            
            # 获取JSON文件所在目录，用于路径转换
            evolution_dir = os.path.dirname(evolution_json_path)
            
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
                    # 转换为相对路径
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
            
            # 添加baseline_info（如果提供了）
            if baseline_info is not None:
                evolution_data["baseline_info"] = baseline_info
            
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
            
            # 重新排序整个JSON结构，确保baseline_info在gp_info上面
            ordered_evolution_data = {}
            if "experiment_info" in evolution_data:
                ordered_evolution_data["experiment_info"] = evolution_data["experiment_info"]
            if "baseline_info" in evolution_data:
                ordered_evolution_data["baseline_info"] = evolution_data["baseline_info"]
            if "gp_info" in evolution_data:
                ordered_evolution_data["gp_info"] = evolution_data["gp_info"]
            # 添加其他可能的字段
            for key, value in evolution_data.items():
                if key not in ["experiment_info", "baseline_info", "gp_info"]:
                    ordered_evolution_data[key] = value
            evolution_data = ordered_evolution_data
            
            # 使用自定义格式化函数保存JSON
            json_str = format_json_compact(evolution_data, indent=2)
            with open(evolution_json_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
    except Exception as e:
        print(f"更新进化过程JSON失败: {e}")

