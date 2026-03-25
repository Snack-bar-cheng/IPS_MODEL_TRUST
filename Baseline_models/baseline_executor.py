"""
Baseline模型执行类
负责加载数据配置、运行baseline模型、保存结果
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

# 过滤 sklearn 的 RuntimeWarning 警告（溢出、无效值、除以零等）
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

from utils.model_creator import create_baseline_model
from utils.evaluator import train_and_evaluate_model
from utils.json_formatter import format_json_compact


class Baseline_Executor:
    """
    Baseline模型执行类
    
    参数:
        data_config_path: 数据配置文件路径（JSON格式）
        random_seeds: 随机种子列表
        model_config: Baseline_Pool返回的模型配置JSON（字典格式）
        cv_folds: 交叉验证折数
    """
    
    def __init__(self, data_config_path: str, random_seeds: List[int], 
                 model_config: Dict, cv_folds: int = 5):
        self.data_config_path = data_config_path
        self.random_seeds = random_seeds if isinstance(random_seeds, list) else [random_seeds]
        self.model_config = model_config
        self.cv_folds = cv_folds
        
        # 获取当前脚本所在目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 加载数据配置
        self._load_data_config()
        
    def _load_data_config(self):
        """加载数据配置文件"""
        # 处理相对路径
        if not os.path.isabs(self.data_config_path):
            config_path = os.path.join(self.base_dir, self.data_config_path)
        else:
            config_path = self.data_config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data_config = json.load(f)
        
        # 提取关键信息
        self.train_set_path = self.data_config['train_set_path']
        self.test_set_path = self.data_config['test_set_path']
        self.selected_feature = self.data_config['selected_feature']
        self.target_column = self.data_config['target_column']
        
        # 处理相对路径
        if not os.path.isabs(self.train_set_path):
            self.train_set_path = os.path.join(self.base_dir, self.train_set_path)
        if not os.path.isabs(self.test_set_path):
            self.test_set_path = os.path.join(self.base_dir, self.test_set_path)
    
    def _load_data(self, target_name: str):
        """
        加载训练集和测试集
        
        参数:
            target_name: 目标列名称（用于提取目标值）
        
        返回:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            X_train: 训练集特征（只使用selected_feature，不包含target）
            X_test: 测试集特征（只使用selected_feature，不包含target）
            y_train: 训练集目标值
            y_test: 测试集目标值
            feature_names: 特征名称列表
        """
        train_df = pd.read_csv(self.train_set_path)
        test_df = pd.read_csv(self.test_set_path)
        
        # 提取特征（只使用selected_feature，确保不包含target）
        feature_cols = [col for col in self.selected_feature if col != target_name]
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        
        # 提取目标值
        y_train = train_df[target_name].values
        y_test = test_df[target_name].values
        
        return train_df, test_df, X_train, X_test, y_train, y_test, feature_cols
    
    def _create_output_folder(self, target_name: str, base_timestamp: str):
        """
        创建输出文件夹（在baseline_result_{timestamp}下创建目标文件夹）
        
        参数:
            target_name: 目标列名称
            base_timestamp: 基础时间戳（用于创建主文件夹）
        
        返回:
            folder_path: 目标文件夹路径（绝对路径）
        """
        # 创建主文件夹
        main_folder_name = f"baseline_result_{base_timestamp}"
        main_folder_path = os.path.join(self.base_dir, main_folder_name)
        os.makedirs(main_folder_path, exist_ok=True)
        
        # 在主文件夹下创建目标文件夹
        target_folder_path = os.path.join(main_folder_path, target_name)
        os.makedirs(target_folder_path, exist_ok=True)
        
        return target_folder_path
    
    def _save_result_json(self, result_data: Dict, folder_path: str, 
                         target_name: str, random_seed: int, timestamp: str):
        """
        保存结果JSON文件
        
        参数:
            result_data: 结果数据字典
            folder_path: 输出文件夹路径
            target_name: 目标列名称（用于日志，不影响文件名）
            random_seed: 随机种子
            timestamp: 时间戳
        """
        filename = f"baseline_{random_seed}_{timestamp}.json"
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def execute(self):
        """
        执行baseline模型评估
        
        返回:
            list: 保存的结果文件路径列表
        """
        # 获取选中的模型列表
        selected_models = self.model_config['selected_models']
        
        # 获取目标列列表
        target_columns = self.target_column if isinstance(self.target_column, list) else [self.target_column]
        
        # 生成基础时间戳（所有目标列共享）
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        # 遍历每个目标列
        for target_name in target_columns:
            print(f"\n处理目标列: {target_name}")
            print("=" * 80)
            
            # 创建该目标列的输出文件夹（在baseline_result_{timestamp}下）
            folder_path = self._create_output_folder(target_name, base_timestamp)
            print(f"结果将保存至: {folder_path}")
            
            # 加载数据（只使用selected_feature作为特征，不包含target）
            train_df, test_df, X_train, X_test, y_train, y_test, feature_names = self._load_data(target_name)
            
            # 获取数据集大小
            train_set_size = X_train.shape[0]
            test_set_size = X_test.shape[0]
            
            print(f"特征数量: {X_train.shape[1]}")
            print(f"训练集样本数: {train_set_size}")
            print(f"测试集样本数: {test_set_size}")
            
            # 获取 all_dataset_path（如果存在）
            all_dataset_path = self.data_config.get('all_dataset_path', '')
            
            # 遍历每个随机种子
            for random_seed in self.random_seeds:
                print(f"\n随机种子: {random_seed}")
                print("-" * 80)
                
                # 记录开始时间
                start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 存储所有模型的结果
                baseline_results = []
                
                # 遍历每个模型
                for model_name in selected_models:
                    print(f"运行模型: {model_name}...")
                    
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
                            cv_folds=self.cv_folds,
                            random_seed=random_seed,
                            feature_names=feature_names
                        )
                        
                        baseline_results.append(result)
                        
                        # 打印结果摘要
                        cv_sum = result.get('cv_metrics', {}).get('cross_validation', {})
                        print(f"  CV_R²: {cv_sum.get('r2_mean', float('nan')):.6f}±{cv_sum.get('r2_std', float('nan')):.6f}")
                        print(f"  Test_R²: {result['test_metrics']['r2']:.6f}")
                        
                    except Exception as e:
                        print(f"  ❌ {model_name} 训练失败: {e}")
                        continue
                
                # 记录结束时间
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 构建结果JSON结构（使用experiment_info格式，类似GP结果）
                result_data = {
                    "experiment_info": {
                        "start_time": start_time,
                        "end_time": end_time,
                        "random_seeds": random_seed,
                        "all_dataset_path": all_dataset_path if all_dataset_path else self.data_config.get('train_set_path', ''),
                        "train_file_path": self.data_config['train_set_path'],
                        "test_file_path": self.data_config['test_set_path'],
                        "train_set_size": train_set_size,
                        "test_set_size": test_set_size,
                        "feature_names": self.selected_feature,
                        "target_variable": target_name
                    },
                    "baseline_info": {
                        "baseline_models": baseline_results
                    }
                }
                
                # 保存结果JSON（文件名格式：baseline_{random_seed}_{timestamp}.json）
                # 使用参考实现的格式化函数
                json_str = format_json_compact(result_data, indent=2)
                
                filepath = os.path.join(folder_path, f"baseline_{random_seed}_{base_timestamp}.json")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                saved_files.append(filepath)
                
                print(f"\n结果已保存至: {filepath}")
                print("-" * 80)
            
            print("=" * 80)
        
        return saved_files

