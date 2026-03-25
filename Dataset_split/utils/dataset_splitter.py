"""
数据集划分类
负责读取数据、划分训练集和测试集、进行KS检验、生成配置文件
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import json

from utils.ks_test import ks_test_for_distribution
from utils.json_utils import format_json_lists_to_single_line


class DatasetSplitter:
    """
    数据集划分类

    参数:
        dataset_path: CSV数据集文件路径
        target_columns: 目标列列表（需要循环处理每一个目标变量）
        delete_columns: 删除列列表
        train_ratio: 训练集比例，默认0.8
        random_seed: 随机种子，默认43
        output_dir: 输出目录，默认为dataset_onfiguration
    """

    def __init__(self, dataset_path, target_columns, delete_columns,
                 train_ratio=0.8, random_seed=43, output_dir="dataset_onfiguration"):
        self.dataset_path = dataset_path
        self.target_columns = target_columns if isinstance(target_columns, list) else [target_columns]
        self.delete_columns = delete_columns if isinstance(delete_columns, list) else ([] if delete_columns is None else [delete_columns])
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.output_dir = output_dir

        _utils_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(_utils_dir)

    def _get_feature_columns(self, all_columns):
        """
        获取特征列（排除目标列和删除列）

        参数:
            all_columns: 所有列名列表

        返回:
            feature_columns: 特征列列表
        """
        exclude_columns = set(self.target_columns)
        exclude_columns.update(self.delete_columns)

        feature_columns = [col for col in all_columns if col not in exclude_columns]
        return feature_columns

    def _create_output_folder(self):
        """
        创建输出文件夹（使用随机种子+时间戳命名）

        返回:
            folder_path: 创建的文件夹路径（相对路径）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.random_seed}_{timestamp}"

        folder_path = os.path.join(self.output_dir, folder_name)
        abs_folder_path = os.path.join(self.base_dir, folder_path)

        os.makedirs(abs_folder_path, exist_ok=True)
        return folder_path

    def _save_train_test_sets(self, train_df, test_df, folder_path, timestamp):
        """
        保存训练集和测试集

        参数:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            folder_path: 输出文件夹路径（相对路径）
            timestamp: 时间戳字符串

        返回:
            train_path: 训练集文件路径（相对路径）
            test_path: 测试集文件路径（相对路径）
        """
        train_filename = f"trainset_{self.random_seed}_{timestamp}.csv"
        test_filename = f"testset_{self.random_seed}_{timestamp}.csv"

        train_path = os.path.join(folder_path, train_filename)
        test_path = os.path.join(folder_path, test_filename)

        abs_train_path = os.path.join(self.base_dir, train_path)
        abs_test_path = os.path.join(self.base_dir, test_path)

        train_df.to_csv(abs_train_path, index=False)
        test_df.to_csv(abs_test_path, index=False)

        return train_path, test_path

    def _generate_config_json(self, all_columns, feature_columns,
                              p_values, folder_path, timestamp, train_path, test_path,
                              ks_plots_pdf_path=None):
        """
        生成配置文件JSON
        """
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        min_p_value = min(p_values) if p_values else 0.0
        avg_p_value = np.mean(p_values) if p_values else 0.0

        abs_dataset_path = os.path.join(self.base_dir, self.dataset_path) if not os.path.isabs(self.dataset_path) else self.dataset_path
        abs_train_path = os.path.join(self.base_dir, train_path) if not os.path.isabs(train_path) else train_path
        abs_test_path = os.path.join(self.base_dir, test_path) if not os.path.isabs(test_path) else test_path

        config = {
            "start_time": start_time,
            "end_time": start_time,
            "all_dataset_path": abs_dataset_path,
            "all_dataset_columns": all_columns,
            "random_seed": self.random_seed,
            "set_ratio": self.train_ratio,
            "min_p_value": float(min_p_value),
            "avg_p_value": float(avg_p_value),
            "selected_feature": feature_columns,
            "target_column": self.target_columns,
            "train_set_path": abs_train_path,
            "test_set_path": abs_test_path,
        }
        if ks_plots_pdf_path:
            config["ks_plots_pdf_path"] = ks_plots_pdf_path

        json_filename = f"data_config_{self.random_seed}_{timestamp}.json"
        config_path = os.path.join(folder_path, json_filename)
        abs_config_path = os.path.join(self.base_dir, config_path)

        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        json_str = format_json_lists_to_single_line(json_str)

        with open(abs_config_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        return config_path, config

    @staticmethod
    def update_config_json_ks_path(config_abs_path, ks_plots_pdf_path):
        """向已存在的 data_config JSON 写入 KS 图 PDF 绝对路径。"""
        with open(config_abs_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config["ks_plots_pdf_path"] = ks_plots_pdf_path
        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        json_str = format_json_lists_to_single_line(json_str)
        with open(config_abs_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    def split_dataset(self):
        """
        执行数据集划分

        生成一个包含所有目标列的训练集和测试集。

        返回:
            dict: abs_output_folder, config_abs_path, config_path, feature_columns,
                  target_columns, train_df, test_df；若中途失败则不返回（抛错）。
        """
        print("=" * 80)
        print("开始数据集划分")
        print("=" * 80)

        dataset_abs_path = os.path.join(self.base_dir, self.dataset_path) if not os.path.isabs(self.dataset_path) else self.dataset_path
        df = pd.read_csv(dataset_abs_path)
        all_columns = list(df.columns)

        missing_targets = [col for col in self.target_columns if col not in all_columns]
        if missing_targets:
            raise ValueError(f"目标列不存在: {missing_targets}")

        missing_delete = [col for col in self.delete_columns if col not in all_columns]
        if missing_delete:
            raise ValueError(f"删除列不存在: {missing_delete}")

        print(f"目标列: {self.target_columns}")
        print(f"删除列: {self.delete_columns}")

        feature_columns = self._get_feature_columns(all_columns)
        print(f"特征列数量: {len(feature_columns)}")

        X = df[feature_columns].values
        y_all = df[self.target_columns].values

        test_size = 1 - self.train_ratio
        X_train, X_test, y_train_all, y_test_all = train_test_split(
            X, y_all, test_size=test_size, random_state=self.random_seed
        )

        print(f"\n训练集: 样本数={X_train.shape[0]}, 特征数={X_train.shape[1]}")
        print(f"测试集: 样本数={X_test.shape[0]}, 特征数={X_test.shape[1]}")

        print("\n进行KS检验，检查训练集和测试集分布一致性...")
        is_consistent, p_values, failed_features = ks_test_for_distribution(
            X_train, X_test, feature_columns, alpha=0.05
        )

        if is_consistent:
            print(f"[OK] 随机种子 {self.random_seed} 通过KS检验，分布一致！")
            print(f"   最小p值: {min(p_values):.6f}")
            print(f"   平均p值: {np.mean(p_values):.6f}")
            print(f"   最大p值: {max(p_values):.6f}")
        else:
            print(f"[FAIL] 随机种子 {self.random_seed} 未通过KS检验")
            print(f"   失败的特征数: {len(failed_features)}/{len(feature_columns)}")
            print(f"   最小p值: {min(p_values):.6f}")
            print(f"   平均p值: {np.mean(p_values):.6f}")
            if len(failed_features) <= 10:
                print(f"   未通过检验的特征: {failed_features}")
            else:
                print(f"   未通过检验的特征（前10个）: {failed_features[:10]}")

        folder_path = self._create_output_folder()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_df = pd.DataFrame(X_train, columns=feature_columns)
        for i, target_col in enumerate(self.target_columns):
            train_df[target_col] = y_train_all[:, i]

        test_df = pd.DataFrame(X_test, columns=feature_columns)
        for i, target_col in enumerate(self.target_columns):
            test_df[target_col] = y_test_all[:, i]

        train_path, test_path = self._save_train_test_sets(
            train_df, test_df, folder_path, timestamp
        )

        print(f"\n训练集已保存至: {train_path}")
        print(f"测试集已保存至: {test_path}")

        config_path, config = self._generate_config_json(
            all_columns, feature_columns, p_values,
            folder_path, timestamp, train_path, test_path,
            ks_plots_pdf_path=None,
        )

        config["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        abs_config_path = os.path.join(self.base_dir, config_path)

        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        json_str = format_json_lists_to_single_line(json_str)
        with open(abs_config_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        print(f"配置文件已保存至: {config_path}")
        print("=" * 80)

        abs_output_folder = os.path.join(self.base_dir, folder_path)
        return {
            "abs_output_folder": abs_output_folder,
            "config_abs_path": abs_config_path,
            "config_path": config_path,
            "feature_columns": feature_columns,
            "target_columns": list(self.target_columns),
            "train_df": train_df,
            "test_df": test_df,
        }
