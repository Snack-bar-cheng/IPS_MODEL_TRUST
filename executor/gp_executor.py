"""
GP执行类
负责执行遗传编程算法
"""

import os
import sys
import json
import random
import logging
import warnings
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd

# 忽略numpy和sklearn的运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
np.seterr(all='ignore')  # 忽略所有numpy的浮点错误

# DEAP相关导入
from deap import base, creator, tools, gp

# 导入本地模块
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from evolution_data_saver.evolution_data import (
    create_evolution_data_structure,
    finalize_evolution_data,
    save_complete_evolution_data
)
from evolution_data_saver.experiment_folder import create_experiment_folder
from evolution_data_saver.dataset_info import (
    create_dataset_info,
    save_dataset_info,
    update_dataset_info_with_ridge_formula
)
from evolution_data_saver.evolution_json import update_evolution_json_with_ridge_formula
from evolution_data_saver.result_writer import write_evo_result
from gp_utils.evaluation import evalSymbReg_Test, evalSymbReg_Test_with_residual_fitting
from gp_utils.ridge_formula import generate_ridge_formula

# 导入baseline相关模块
baseline_path = os.path.join(os.path.dirname(__file__), '..', 'Baseline_models')
if baseline_path not in sys.path:
    sys.path.insert(0, baseline_path)
from utils.model_creator import create_baseline_model
from utils.evaluator import train_and_evaluate_model

# 导入参考实现（用于某些工具函数）
reference_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gps_lr_llm_step_residual')
if reference_path not in sys.path:
    sys.path.insert(0, reference_path)

# 导入本地模块
from .gp_system_setup import setup_primitive_set, setup_toolbox
from .gp_evolution import run_evolution
from .gp_data_loader import load_data, initialize_population
from .gp_result_saver import save_results

# 定义类型系统
Float1 = float
Vector1 = np.ndarray

# 创建类型系统
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


class GPExecutor:
    """GP执行类"""
    
    def __init__(self, data_config_path: str, gp_config: dict, random_seeds: List[int], output_dir_name: Optional[str] = None):
        """
        初始化GP执行器
        
        参数:
            data_config_path: 数据配置文件路径（JSON格式）
            gp_config: GP配置字典（由GPConfigBuilder生成）
            random_seeds: 随机种子列表
            output_dir_name: 输出目录名称（可选），如果提供则使用此名称，否则使用时间戳格式
        """
        self.data_config_path = data_config_path
        self.gp_config = gp_config
        self.random_seeds = random_seeds
        self.output_dir_name = output_dir_name
        # 保存配置名称（用于保存到JSON），如果没有则为None
        self.config_name = output_dir_name if output_dir_name else None
        
        # 加载数据配置
        self.data_config = self._load_data_config()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _resolve_dataset_csv_path(self, path_value: str) -> str:
        """
        将 train/test CSV 路径解析为实际存在的文件。
        JSON 中常残留其它机器上的绝对路径；若文件不存在，则回退为与 data_config 同目录下的同名文件。
        """
        if not path_value:
            return path_value
        base_dir = os.path.dirname(os.path.abspath(self.data_config_path))
        if os.path.isfile(path_value):
            return os.path.normpath(os.path.abspath(path_value))
        # 与配置文件同目录：basename 或相对路径
        candidates = [
            os.path.join(base_dir, os.path.basename(path_value)),
            os.path.join(base_dir, path_value) if not os.path.isabs(path_value) else None,
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return os.path.normpath(os.path.abspath(c))
        return path_value

    def _load_data_config(self) -> dict:
        """加载数据配置文件，并修正 train/test CSV 路径（支持相对路径、同目录回退）。"""
        with open(self.data_config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        for key in ("train_set_path", "test_set_path"):
            if key in cfg and cfg[key]:
                cfg[key] = self._resolve_dataset_csv_path(cfg[key])
        return cfg
    
    def execute(self) -> List[str]:
        """
        执行GP算法
        
        返回:
            List[str]: 生成的JSON文件路径列表
        """
        saved_files = []
        target_columns = self.data_config.get('target_column', [])
        
        # 如果target_column是字符串，转换为列表
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        # 创建统一的输出目录（所有目标列和随机种子共享）
        # 生成时间戳（用于文件名，无论是否使用自定义目录名）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        base_dir = os.path.dirname(__file__)
        if self.output_dir_name:
            # 使用提供的目录名称，清理特殊字符以确保目录名有效
            import re
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', self.output_dir_name)
            safe_name = safe_name.strip()
            output_dir = os.path.join(base_dir, '..', safe_name)
        else:
            # 使用时间戳格式（默认行为）
            output_dir = os.path.join(base_dir, '..', 'gps_result_' + timestamp)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历每个目标列
        for target_name in target_columns:
            self.logger.info(f"开始处理目标列: {target_name}")
            
            # 创建目标列子目录
            target_dir = os.path.join(output_dir, target_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # 加载数据
            X_train, y_train, X_test, y_test, feature_names = load_data(
                self.data_config, target_name
            )
            
            # 遍历每个随机种子
            for random_seed in self.random_seeds:
                self.logger.info(f"  随机种子: {random_seed}")
                
                # 执行单次GP运行
                result_file = self._run_single_gp(
                    X_train, y_train, X_test, y_test, 
                    feature_names, target_name, random_seed,
                    target_dir, timestamp
                )
                
                if result_file:
                    saved_files.append(result_file)
        
        return saved_files
    
    def _run_single_gp(self, X_train, y_train, X_test, y_test, 
                      feature_names, target_name, random_seed,
                      target_dir, timestamp):
        """执行单次GP运行"""
        # ========== 设置随机种子（确保可复现性） ==========
        # 设置Python标准库random模块的随机种子
        random.seed(random_seed)
        # 设置numpy的随机种子
        np.random.seed(random_seed)
        
        shap_cfg = self.gp_config.get("shap", {})
        shap_open = shap_cfg.get("open", self.gp_config.get("shap_open", False))
        shap_bg_size = shap_cfg.get("background_sample_size")
        shap_explain_size = shap_cfg.get("explain_sample_size")
        shap_ks_threshold = shap_cfg.get("ks_pvalue_threshold", 0.05)
        shap_max_attempts = shap_cfg.get("max_attempts", 5)
        evolution_process_dir = os.path.join(target_dir, target_name, "evolution_process")
        shap_save_dir = os.path.join(evolution_process_dir, "shap_output")
        if shap_open:
            os.makedirs(shap_save_dir, exist_ok=True)
        
        # 先运行baseline，后续再执行GP；运行后重置随机种子确保GP复现性
        self.logger.info(f"[SHAP][DEBUG] baseline阶段配置: shap_open={shap_open}, shap_save_dir={shap_save_dir}")
        # 构建实际的baseline保存目录：target_dir/target_name（与evolution_process_dir同级）
        baseline_save_dir = os.path.join(target_dir, target_name)
        baseline_info = self._run_baseline_evaluation(
            X_train, y_train, X_test, y_test,
            feature_names, target_name, random_seed,
            target_dir=baseline_save_dir,  # 传递实际的保存目录用于保存可视化文件
            shap_open=shap_open,
            shap_save_dir=shap_save_dir,
            shap_bg_size=shap_bg_size,
            shap_explain_size=shap_explain_size,
            shap_ks_threshold=shap_ks_threshold,
            shap_max_attempts=shap_max_attempts
        )
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.logger.info(f"已设置随机种子: {random_seed} (确保实验可复现)")
        
        # 设置GP系统
        func_config = self.gp_config['function_set']
        high_func_config = self.gp_config['high_function']
        pset = setup_primitive_set(X_train, feature_names, func_config, high_func_config)
        
        # 加载LLM特征
        llm_features = self._load_llm_features()
        
        # 设置工具箱
        toolbox = setup_toolbox(
            pset, X_train, y_train, self.gp_config, 
            feature_names, llm_features
        )
        
        # 创建进化数据结构
        evolution_data = self._create_evolution_data(
            X_train, y_train, X_test, y_test, 
            feature_names, target_name, random_seed
        )
        
        # 初始化种群
        population = initialize_population(toolbox, self.gp_config, llm_features)
        
        # 统计并输出种群初始化信息
        population_size = self.gp_config['population_size']
        init_config = self.gp_config['initialization']
        population_init_info = {}  # 用于保存到txt文件
        
        if init_config['llm']['enabled'] and llm_features and hasattr(toolbox, 'individual_llm'):
            llm_ratio = init_config['llm']['ratio']
            desired_llm_count = int(population_size * llm_ratio)
            actual_llm_count = min(desired_llm_count, len(llm_features))
            random_count = population_size - actual_llm_count
            self.logger.info(f"初始化种群: 采用 {actual_llm_count} 个LLM特征, 生成 {random_count} 个random特征")
            population_init_info['llm_count'] = actual_llm_count
            population_init_info['random_count'] = random_count
        else:
            self.logger.info(f"初始化种群: 生成 {population_size} 个random特征（未启用LLM特征）")
            population_init_info['llm_count'] = 0
            population_init_info['random_count'] = population_size
        
        # 统计每个High函数的使用次数（如果启用了High函数）
        high_func_config = self.gp_config['high_function']
        enable_high = high_func_config.get('enable', False)
        if enable_high:
            high_usage_count = {}
            for ind in population:
                ind_expr = list(ind)
                if len(ind_expr) > 0:
                    root_node = ind_expr[0]
                    if hasattr(root_node, 'name') and root_node.name and root_node.name.startswith('High_'):
                        high_name = root_node.name
                        high_usage_count[high_name] = high_usage_count.get(high_name, 0) + 1
            
            # 输出每个High函数的使用统计
            if high_usage_count:
                for high_name in sorted(high_usage_count.keys(), key=lambda x: int(x.split('_')[1])):
                    count = high_usage_count[high_name]
                    self.logger.info(f"初始化后{high_name}的个体有 {count} 个")
                population_init_info['high_usage'] = high_usage_count
            else:
                self.logger.warning("⚠️ 警告：初始化后没有找到任何使用High函数的个体！")
                population_init_info['high_usage'] = {}
        
        # 执行进化
        enable_dynamic_expansion = high_func_config.get('enable_dynamic_expansion', False)
        enable_high = high_func_config.get('enable', False)
        
        pset_for_evolution = pset if (enable_high and enable_dynamic_expansion) else None
        
        population, logbook, training_duration, dynamic_expansion_logs = run_evolution(
            population, toolbox, self.gp_config, evolution_data,
            X_train, y_train, feature_names, target_name,
            pset=pset_for_evolution, llm_features=llm_features
        )
        
        # 记录训练时间（核心训练时间：只包括进化过程，不包括数据加载、系统设置、结果保存等）
        self.logger.info("=" * 80)
        self.logger.info(f"核心训练时间统计:")
        self.logger.info(f"  训练时间: {training_duration:.4f} 秒")
        self.logger.info(f"  训练时间: {training_duration/60:.2f} 分钟")
        self.logger.info(f"  训练时间: {training_duration/3600:.4f} 小时")
        self.logger.info("=" * 80)
        
        # 最终化进化数据
        finalize_evolution_data(evolution_data)
        
        # ========== 最终模型评估 ==========
        # 从种群中获取最佳个体（适应度最高的个体）
        if len(population) == 0:
            self.logger.warning("种群为空，无法进行最终评估")
            return None
        best_individual = max(population, key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
        if not best_individual.fitness.valid:
            self.logger.warning("最佳个体适应度无效，无法进行最终评估")
            return None
        
        self.logger.info("计算最终模型指标...")
        # 判断是否使用High函数模式
        enable_high = self.gp_config['high_function'].get('enable', False)
        test_metrics, train_metrics, model, gp_expr = evalSymbReg_Test(
            best_individual, toolbox, X_train, y_train, X_test, y_test,
            enable_high=enable_high
        )
        
        train_r2 = train_metrics['r2']
        train_mse = train_metrics['mse']
        train_rmse = train_metrics['rmse']
        train_mae = train_metrics['mae']
        
        test_r2 = test_metrics['r2']
        test_mse = test_metrics['mse']
        test_rmse = test_metrics['rmse']
        test_mae = test_metrics['mae']
        
        train_model = model
        test_model = model
        
        # 生成Ridge回归公式（仅High函数模式）
        ridge_formula = None
        if enable_high and test_model is not None:
            ridge_formula = generate_ridge_formula(test_model, gp_expr, target_name=target_name)
            self.logger.info(f"Ridge回归公式: {ridge_formula}")
        else:
            self.logger.info("传统GP模式：不使用Ridge回归公式")
        
        train_metrics = {
            "r2": float(train_r2),
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "mae": float(train_mae),
        }
        test_metrics = {
            "r2": float(test_r2),
            "mse": float(test_mse),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
        }
        
        # ========== 残差拟合评估 ==========
        # 初始化残差拟合结果变量（用于保存到JSON）
        residual_fitting_results = {"original": {}, "with_residual": []}
        residual_training_duration = None  # 初始化残差训练时间变量
        residual_config = self.gp_config.get('residual_fitting', {})
        if residual_config.get('enable', False):
            self.logger.info("=" * 80)
            
            # 获取模型列表（兼容旧的配置格式）
            residual_models = residual_config.get('models', None)
            if residual_models is None:
                # 兼容旧的配置格式：使用 'model' 字段
                model_type = residual_config.get('model', 'rf')
                if model_type == 'rf':
                    residual_models = ['RandomForest']
                else:
                    residual_models = ['RidgeCV']
            
            self.logger.info(f"残差拟合评估（使用模型: {', '.join(residual_models)}）...")
            # 判断是否使用High函数模式（与最终模型评估保持一致）
            enable_high = self.gp_config['high_function'].get('enable', False)
            try:
                residual_results = evalSymbReg_Test_with_residual_fitting(
                    best_individual, toolbox, X_train, y_train, X_test, y_test, 
                    residual_models=residual_models,
                    random_seed=random_seed,
                    enable_high=enable_high,
                    shap_open=shap_open,
                    shap_save_dir=shap_save_dir,
                    feature_names=feature_names,
                    target_name=target_name,
                    shap_bg_size=shap_bg_size,
                    shap_explain_size=shap_explain_size,
                    shap_ks_threshold=shap_ks_threshold,
                    shap_max_attempts=shap_max_attempts
                )
                
                # 为JSON保存准备残差拟合结果，添加特征名称信息
                residual_fitting_results = residual_results.copy() if residual_results is not None else {"original": {}, "with_residual": []}
                original_train = residual_results['original']['train']
                original_test = residual_results['original']['test']
                
                # 处理多个模型的结果
                with_residual_list = residual_results.get('with_residual', [])
                
                # 为每个模型添加特征名称信息
                for residual_item in with_residual_list:
                    residual_contribution = residual_item.get('residual_contribution', {})
                    if 'feature_importances' in residual_contribution and feature_names:
                        # 添加特征名称映射
                        residual_contribution['feature_names'] = feature_names
                
                # 输出日志
                self.logger.info("-" * 80)
                self.logger.info("残差拟合对比结果:")
                self.logger.info("")
                self.logger.info("【原始模型（RidgeCV）】")
                self.logger.info("  训练集指标:")
                self.logger.info(f"    R²:  {original_train['r2']:.6f}")
                self.logger.info(f"    MSE: {original_train['mse']:.6f}")
                self.logger.info(f"    RMSE: {original_train['rmse']:.6f}")
                self.logger.info(f"    MAE:  {original_train['mae']:.6f}")
                self.logger.info("  测试集指标:")
                self.logger.info(f"    R²:  {original_test['r2']:.6f}")
                self.logger.info(f"    MSE: {original_test['mse']:.6f}")
                self.logger.info(f"    RMSE: {original_test['rmse']:.6f}")
                self.logger.info(f"    MAE:  {original_test['mae']:.6f}")
                self.logger.info("")
                
                # 对每个残差模型输出结果
                for residual_item in with_residual_list:
                    model_name = residual_item.get('model_name', 'Unknown')
                    residual_train = residual_item.get('train', {})
                    residual_test = residual_item.get('test', {})
                    residual_contribution = residual_item.get('residual_contribution', {})
                    residual_training_duration = residual_item.get('residual_training_duration', None)
                    
                    self.logger.info(f"【残差拟合后（{model_name}）】")
                    if residual_training_duration is not None:
                        self.logger.info(f"  训练时间: {residual_training_duration:.4f} 秒 ({residual_training_duration/60:.2f} 分钟)")
                    self.logger.info("  训练集指标:")
                    self.logger.info(f"    R²:  {residual_train.get('r2', 0):.6f}")
                    self.logger.info(f"    MSE: {residual_train.get('mse', 0):.6f}")
                    self.logger.info(f"    RMSE: {residual_train.get('rmse', 0):.6f}")
                    self.logger.info(f"    MAE:  {residual_train.get('mae', 0):.6f}")
                    self.logger.info("  测试集指标:")
                    self.logger.info(f"    R²:  {residual_test.get('r2', 0):.6f}")
                    self.logger.info(f"    MSE: {residual_test.get('mse', 0):.6f}")
                    self.logger.info(f"    RMSE: {residual_test.get('rmse', 0):.6f}")
                    self.logger.info(f"    MAE:  {residual_test.get('mae', 0):.6f}")
                    
                    # 改进情况
                    train_r2_improvement = residual_train.get('r2', 0) - original_train['r2']
                    test_r2_improvement = residual_test.get('r2', 0) - original_test['r2']
                    self.logger.info("  改进情况:")
                    self.logger.info(f"    训练集R²改进:  {train_r2_improvement:+.6f} ({train_r2_improvement/original_train['r2']*100:+.2f}%)" if original_train['r2'] != 0 else f"    训练集R²改进:  {train_r2_improvement:+.6f}")
                    self.logger.info(f"    测试集R²改进:  {test_r2_improvement:+.6f} ({test_r2_improvement/original_test['r2']*100:+.2f}%)" if original_test['r2'] != 0 else f"    测试集R²改进:  {test_r2_improvement:+.6f}")
                    
                    # 特征重要性（如果有）
                    if residual_contribution and 'feature_importances' in residual_contribution:
                        self.logger.info("")
                        self.logger.info(f"  【{model_name}残差贡献情况】")
                        self.logger.info("  注意：残差模型使用原始数据集的全部特征，与原始模型（使用GP高阶特征）不同")
                        self.logger.info(f"  残差模型训练集R²: {residual_contribution.get('residual_train_r2', 0):.6f}")
                        self.logger.info(f"  残差模型测试集R²: {residual_contribution.get('residual_test_r2', 0):.6f}")
                        
                        feature_importances = residual_contribution.get('feature_importances', [])
                        if isinstance(feature_importances, list) and len(feature_importances) > 0:
                            # SHAP格式：列表中是dict
                            if isinstance(feature_importances[0], dict):
                                self.logger.info("  【各特征在残差中的贡献（按mean_abs_shap排序，前5名）】")
                                sorted_imps = sorted(feature_importances, key=lambda x: x.get("mean_abs_shap", 0), reverse=True)
                                for idx, item in enumerate(sorted_imps[:5]):
                                    name = item.get("feature_name", f"feat_{idx}")
                                    ma = item.get("mean_abs_shap", 0)
                                    ms = item.get("mean_shap", 0)
                                    self.logger.info(f"    {idx+1:2d}. {name:30s}: mean_abs_shap={ma:.6f}, mean_shap={ms:.6f}")
                            else:
                                # 旧格式：纯数值列表
                                self.logger.info("  【各特征在残差中的贡献（按重要性排序，前5名）】")
                                sorted_importances = sorted(enumerate(feature_importances), key=lambda x: x[1], reverse=True)
                                total_importance = sum(feature_importances)
                                for idx, (feat_idx, importance) in enumerate(sorted_importances[:5]):
                                    if feat_idx < len(feature_names):
                                        feat_name = feature_names[feat_idx]
                                    else:
                                        feat_name = f"特征_{feat_idx}"
                                    percentage = (importance / total_importance * 100) if total_importance > 0 else 0
                                    self.logger.info(f"    {idx+1:2d}. {feat_name:30s}: {importance:.6f} ({percentage:5.2f}%)")
                    
                    self.logger.info("")
                
                self.logger.info("-" * 80)
                self.logger.info("=" * 80)
            except Exception as e:
                self.logger.warning(f"残差拟合评估失败: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                residual_fitting_results = {"original": {}, "with_residual": [], "error": str(e)}
        
        # ========== 创建实验文件夹 ==========
        # 创建配置对象
        class Config:
            def __init__(self, data_config, result_root):
                self.data_file = data_config.get('all_dataset_path', '')
                self.result_root = result_root
                self.dataset_name = "dataset"
        
        config = Config(self.data_config, target_dir)
        experiment_dir = create_experiment_folder(target_dir, random_seed, target_name=target_name)
        
        # ========== 保存数据集信息 ==========
        dataset_info = create_dataset_info(
            self.data_config, X_train, y_train, X_test, y_test, feature_names
        )
        dataset_info_path = save_dataset_info(dataset_info, experiment_dir, target_name=target_name)
        
        # 更新数据集信息，添加Ridge公式
        update_dataset_info_with_ridge_formula(
            dataset_info_path, ridge_formula, train_metrics, test_metrics
        )
        
        # ========== 保存进化数据 ==========
        save_complete_evolution_data(
            evolution_data, target_dir, "dataset", random_seed, target_name=target_name
        )
        evolution_json_path = os.path.join(
            experiment_dir, 'evolution_process', 
            f"{random_seed}_{target_name}.json" if target_name else f"{random_seed}.json"
        )
        
        # ========== 执行Baseline评估 ==========
        # ========== 更新进化JSON ==========
        # 获取CV指标
        cv_metrics = getattr(best_individual, 'cv_metrics', None)
        
        # 调试：检查残差拟合结果
        if residual_fitting_results is not None:
            self.logger.info(f"准备保存残差拟合结果到JSON，结果类型: {type(residual_fitting_results)}")
            if isinstance(residual_fitting_results, dict):
                self.logger.info(f"残差拟合结果包含的键: {list(residual_fitting_results.keys())}")
        
        # 更新进化JSON，添加Ridge公式、残差拟合结果、baseline结果等
        update_evolution_json_with_ridge_formula(
            evolution_json_path=evolution_json_path,
            ridge_formula=ridge_formula,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            best_expression=gp_expr,
            cv_metrics=cv_metrics,
            gp_hyperparams=self._get_gp_hyperparams(),
            split_dataset_paths=None,
            residual_fitting_results=residual_fitting_results,
            training_duration=training_duration,  # 传递训练时间
            residual_training_duration=residual_training_duration,  # 传递残差模型训练时间
            baseline_info=baseline_info  # 传递baseline结果
        )
        
        # ========== 写入结果文本文件 ==========
        try:
            csv_path = self.data_config.get('all_dataset_path', '')
            # 将txt文件保存到experiment_dir（GP结果目录）中
            evo_file = write_evo_result(
                csv_path=csv_path,
                feature_names=feature_names,
                target_name=target_name,
                dropped_columns=self.data_config.get('delete_columns', []),  # 从data_config中获取删除的列
                seed=random_seed,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                expression=gp_expr,
                run_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                evolution_json_path=evolution_json_path,
                ridge_formula=ridge_formula,
                cv_metrics=cv_metrics,
                output_dir=experiment_dir,  # 指定输出目录为GP结果目录
                residual_fitting_results=residual_fitting_results,  # 传递残差拟合结果
                training_duration=training_duration,  # 传递训练时间
                residual_training_duration=residual_training_duration,  # 传递残差模型训练时间
                population_init_info=population_init_info,  # 传递种群初始化统计信息
                dynamic_expansion_logs=dynamic_expansion_logs,  # 传递动态分支扩展日志
                train_file_path=self.data_config.get('train_set_path'),  # 传递训练集路径
                test_file_path=self.data_config.get('test_set_path'),  # 传递测试集路径
            )
            self.logger.info(f"结果已追加到: {evo_file}")
        except Exception as e:
            self.logger.warning(f"写入 evo_result.txt 失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # ========== 树可视化 ==========
        llm_features_for_viz = self._load_llm_features()
        tree_save_dir = os.path.join(experiment_dir, 'best_tree_visualization')
        os.makedirs(tree_save_dir, exist_ok=True)
        
        try:
            from evolution_visualization import Plot_tree
            Plot_tree(
                best_individual, random_seed, tree_save_dir,
                feature_names, 0,
                target_name=target_name,
                llm_features=llm_features_for_viz,
                is_last_generation=True,
                ridge_formula=ridge_formula
            )
        except Exception as e:
            self.logger.warning(f"树可视化失败: {e}")
        
        return evolution_json_path
    
    def _load_llm_features(self):
        """加载LLM特征"""
        llm_features = []
        init_config = self.gp_config['initialization']
        
        if init_config['llm']['enabled']:
            llm_paths = init_config['llm']['llm_feature_paths']
            if llm_paths:
                utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
                if utils_path not in sys.path:
                    sys.path.insert(0, utils_path)
                from llm_new_feature.llm_to_gp_converter import load_llm_features
                llm_features = load_llm_features(json_file_paths=llm_paths)
        
        return llm_features
    
    def _create_evolution_data(self, X_train, y_train, X_test, y_test,
                              feature_names, target_name, random_seed):
        """创建进化数据结构"""
        # 使用统一的GP超参数获取方法
        gp_hyperparams = self._get_gp_hyperparams()
        
        # 创建配置对象（简化版）
        class Config:
            def __init__(self, data_config):
                self.data_file = data_config.get('all_dataset_path', '')
                self.result_root = os.path.dirname(__file__)
                self.dataset_name = "dataset"
        
        config = Config(self.data_config)
        
        evolution_data = create_evolution_data_structure(
            dataset_name="dataset",
            random_seeds=random_seed,
            ngen=self.gp_config['generations'],
            population_size=self.gp_config['population_size'],
            config=config,
            all_dataset_path=self.data_config.get('all_dataset_path'),
            train_file_path=self.data_config.get('train_set_path'),
            test_file_path=self.data_config.get('test_set_path'),
            train_set_size=len(y_train),
            test_set_size=len(y_test),
            feature_names=feature_names,
            target_variable=target_name,
            gp_hyperparams=gp_hyperparams
        )
        
        return evolution_data
    
    def _get_gp_hyperparams(self):
        """获取GP超参数字典"""
        gp_hyperparams = {
            "name": self.config_name if self.config_name else None,
            "population_size": self.gp_config['population_size'],
            "generations": self.gp_config['generations'],
            "cx_prob": self.gp_config['crossover']['cx_prob'],
            "mut_prob": self.gp_config['mutation']['mut_prob'],
            "elitism_prob": self.gp_config['elitism_prob'],
            "initial_min_height": self.gp_config['initialization']['random'].get('initial_min_height', self.gp_config['initialization']['random'].get('initial_min_depth', 1)),
            "initial_max_height": self.gp_config['initialization']['random'].get('initial_max_height', self.gp_config['initialization']['random'].get('initial_max_depth', 2)),
            "max_tree_height": self.gp_config.get('max_tree_height', 4),  # 使用全局配置
            "mutate_gen_full_min": self.gp_config['mutation']['mutate_gen_full_min'],
            "mutate_gen_full_max": self.gp_config['mutation']['mutate_gen_full_max'],
            "tournament_size": self.gp_config['selection']['tournament_size'],
            "hof_size": self.gp_config['selection']['hof_size'],
            "cv_folds": self.gp_config['cv_folds'],
            "scale_factor": self.gp_config['scale_factor'],
            "llm_init_enabled": self.gp_config['initialization']['llm']['enabled'],
            "llm_init_ratio": self.gp_config['initialization']['llm']['ratio'],
            "llm_max_tree_height": self.gp_config['initialization']['llm'].get('llm_max_tree_height', self.gp_config['initialization']['llm'].get('llm_max_tree_depth', 2)),
            "llm_feature_paths": self.gp_config['initialization']['llm']['llm_feature_paths'],
        }
        
        # 添加选择策略相关参数
        selection_config = self.gp_config['selection']
        gp_hyperparams["selection_strategy"] = selection_config.get('strategy', 'tournament')
        gp_hyperparams["enable_gwo"] = selection_config.get('enable_gwo', False)
        gp_hyperparams["gwo_type"] = selection_config.get('gwo_type', 'traditional')
        
        # 添加GWO配置参数（如果启用）
        if selection_config.get('enable_gwo', False) and 'gwo_config' in selection_config:
            gwo_config = selection_config['gwo_config']
            gp_hyperparams["gwo_config"] = {
                "tau1": gwo_config.get('tau1', 0.45),
                "tau2": gwo_config.get('tau2', 0.65),
                "a0": gwo_config.get('a0', 2.0),
                "am": gwo_config.get('am', 1.0),
                "af": gwo_config.get('af', 0.0),
                "rho_alpha": gwo_config.get('rho_alpha', 0.1),
                "rho_beta": gwo_config.get('rho_beta', 0.1),
                "rho_delta": gwo_config.get('rho_delta', 0.1),
                "use_adaptive_layers": gwo_config.get('use_adaptive_layers', False),
                "diversity_threshold": gwo_config.get('diversity_threshold', 0.15),
                "adaptive_factor": gwo_config.get('adaptive_factor', 0.1)
        }
        
        # 添加High函数相关参数
        high_func_config = self.gp_config['high_function']
        gp_hyperparams["enable_high_function"] = high_func_config.get('enable', False)
        gp_hyperparams["enable_dynamic_expansion"] = high_func_config.get('enable_dynamic_expansion', False)
        if high_func_config.get('enable', False):
            gp_hyperparams["base_high_n"] = high_func_config.get('base_high_n', 3)
            gp_hyperparams["expansion_interval"] = high_func_config.get('expansion_interval', 4)
            gp_hyperparams["max_high_n"] = high_func_config.get('max_high_n', 10)
            # 添加其他High函数配置参数
            if 'expansion_llm_ratio' in high_func_config:
                gp_hyperparams["expansion_llm_ratio"] = high_func_config.get('expansion_llm_ratio', 0)
            if 'open_dynamic' in high_func_config:
                gp_hyperparams["open_dynamic"] = high_func_config.get('open_dynamic', False)
            if 'growth_threshold' in high_func_config:
                gp_hyperparams["growth_threshold"] = high_func_config.get('growth_threshold', 1.0)
            if 'window_size' in high_func_config:
                gp_hyperparams["window_size"] = high_func_config.get('window_size', 3)
            if 'expansion_min_height' in high_func_config:
                gp_hyperparams["expansion_min_height"] = high_func_config.get('expansion_min_height', 0)
            if 'expansion_max_height' in high_func_config:
                gp_hyperparams["expansion_max_height"] = high_func_config.get('expansion_max_height', 1)
        
        return gp_hyperparams
    
    def _run_baseline_evaluation(self, X_train, y_train, X_test, y_test, 
                                  feature_names, target_name, random_seed,
                                  target_dir=None,
                                  shap_open=False, shap_save_dir=None,
                                  shap_bg_size=None, shap_explain_size=None,
                                  shap_ks_threshold=0.05, shap_max_attempts=5):
        """
        执行baseline模型评估
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            feature_names: 特征名称列表
            target_name: 目标变量名称
            random_seed: 随机种子
            target_dir: 目标目录路径（用于保存可视化文件）
        
        返回:
            dict: baseline_info字典，格式为 {"baseline_models": [...]}，如果未启用则返回None
        """
        # 检查baseline配置
        baseline_config = self.gp_config.get('baseline', {})
        if not baseline_config.get('enable', False):
            return None
        
        model_names = baseline_config.get('model_names', [])
        if not model_names:
            self.logger.warning("Baseline已启用但未指定模型，跳过baseline评估")
            return None
        
        self.logger.info(f"开始执行baseline评估，模型列表: {model_names}")
        
        baseline_results = []
        cv_folds = self.gp_config.get('cv_folds', 5)
        
        # 遍历每个模型
        for model_name in model_names:
            self.logger.info(f"运行baseline模型: {model_name}...")
            
            try:
                # 创建模型
                model = create_baseline_model(
                    model_name=model_name,
                    input_size=X_train.shape[1],
                    random_seed=random_seed
                )
                
                # 训练和评估模型
                result = train_and_evaluate_model(
                    model=model,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    cv_folds=cv_folds,
                    random_seed=random_seed,
                    feature_names=feature_names,
                    shap_open=shap_open,
                    shap_save_dir=shap_save_dir,
                    target_name=target_name,
                    shap_bg_size=shap_bg_size,
                    shap_explain_size=shap_explain_size,
                    shap_ks_threshold=shap_ks_threshold,
                    shap_max_attempts=shap_max_attempts,
                    target_dir=target_dir  # 传递target_dir用于保存可视化文件
                )
                
                baseline_results.append(result)
                
                # 打印结果摘要
                cv_sum = result.get('cv_metrics', {}).get('cross_validation', {})
                self.logger.info(f"  {model_name} - CV_R²: {cv_sum.get('r2_mean', float('nan')):.6f}±{cv_sum.get('r2_std', float('nan')):.6f}")
                self.logger.info(f"  {model_name} - Test_R²: {result['test_metrics']['r2']:.6f}")
                
            except Exception as e:
                self.logger.warning(f"  ❌ {model_name} 训练失败: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                continue
        
        if not baseline_results:
            self.logger.warning("所有baseline模型评估失败")
            return None
        
        baseline_info = {
            "baseline_models": baseline_results
        }
        
        self.logger.info(f"Baseline评估完成，共评估 {len(baseline_results)} 个模型")
        return baseline_info
