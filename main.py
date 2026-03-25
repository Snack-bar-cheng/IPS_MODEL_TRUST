"""
GPS主程序
只包含主要调用逻辑
"""

import os
import warnings
import numpy as np

# 忽略numpy和sklearn的运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
# 忽略训练/预测列名不一致的提示
warnings.filterwarnings('ignore', category=UserWarning, message='.*does not have valid feature names.*')
np.seterr(all='ignore')  # 忽略所有numpy的浮点错误

from gp_config_builder import GPConfigBuilder
from executor import GPExecutor


def main():
    """主函数"""
    _root = os.path.dirname(os.path.abspath(__file__))

    # 1. 数据配置文件路径（所有配置共用，相对本文件所在项目根目录）
    data_config_path = os.path.join(
        _root,
        "Dataset_split",
        "dataset_onfiguration",
        "43_20251203_235329",
        "data_config_43_20251203_235329.json",
    )

    # 2. 随机种子列表（所有配置共用）
    random_seeds = list(range(1, 11))  # 1-10的随机种子列表

    llm_feature_path = [
        os.path.join(
            _root,
            "utils",
            "llm_new_feature",
            "json_save",
            "llm_Ash_Deformation_20251201_130238.json",
        )
    ]

    random_seeds = list(range(1, 2))
    max_tree_height = 2

    gp_config_item =        {
                # 种群与迭代
                "population_size": 100,
                "generations": 50,
                "elitism_prob": 0.03,
                "cv_folds": 5,
                "scale_factor": 100,
                "max_tree_height": max_tree_height,

                # 初始化策略
                "initialization": {
                    "random": {
                        "enabled": True,
                        "ratio": 1,
                        "initial_min_height": 1,
                        "initial_max_height": 2
                    },
                    "llm": {
                        "enabled": True,
                        "ratio": 0,
                        "llm_max_tree_height": 1,
                        "llm_feature_path": llm_feature_path
                    }
                },

                # 选择策略
                "selection": {
                    "strategy": "tournament",
                    "tournament_size": 7,
                    "hof_size": 20,
                    # GWO算法开关
                    "enable_gwo": False,  # 设置为True启用GWO算法，False使用锦标赛选择
                    # GWO算法类型选择
                    "gwo_type": "enhanced",  # "traditional" 使用传统GWO, "enhanced" 使用改进的GWO（默认）
                    # GWO算法配置（仅在enable_gwo=True且gwo_type="enhanced"时生效）
                    "gwo_config": {
                        # 阶段阈值（针对50代优化，已根据实验结果优化）
                        "tau1": 0.45,  # 探索阶段结束比例 (25% = 12-13代) - 增加早期探索
                        "tau2": 0.65,  # 平衡阶段结束比例 (65% = 32-33代) - 延长平衡期
                        # 收敛因子参数（优化后）
                        "a0": 2.2,    # 初始收敛因子（探索阶段）- 增强早期探索能力
                        "am": 1.2,    # 中期收敛因子（平衡阶段）- 保持适度探索
                        "af": 0.15,   # 最终收敛因子（开发阶段）- 保持最小探索避免过早收敛
                        # 层比例参数（优化后，增加精英引导）
                        "rho_alpha": 0.08,  # Alpha层比例（最佳个体）- 增加精英保留
                        "rho_beta": 0.12,   # Beta层比例（次佳个体）- 增强引导
                        "rho_delta": 0.15,  # Delta层比例（第三佳个体）
                        # 自适应参数（优化后）
                        "use_adaptive_layers": True,
                        "diversity_threshold": 0.12,  # 降低阈值，更早触发自适应调整
                        "adaptive_factor": 0.15  # 增加调整幅度，更快响应多样性变化
                    }
                },

                # 交叉策略
                "crossover": {
                    "cx_prob": 0.8,
                    "strategy": "one_point",
                },

                # 变异策略
                "mutation": {
                    "mut_prob": 0.17,
                    "strategy": "uniform",
                    "mutate_gen_full_min": 0,
                    "mutate_gen_full_max": 1,
                },

                # High函数配置
                "high_function": {
                    "enable": True,
                    "high_n": [1,2,3,4,5,6,7,8,9,10],
                    "enable_dynamic_expansion": True,
                    "base_high_n": 1,
                    "expansion_interval": 3,
                    "max_high_n": 10,
                    "expansion_llm_ratio": 0.5,
                    "open_dynamic": False,          # 是否启用自适应扩展策略
                    "growth_threshold": 1.0,       # 精英平均fitness跨窗口平均增长阈值（R2*100刻度）
                    "window_size": 3,              # 滑动窗口大小（通常与expansion_interval一致）
                    "expansion_min_height": 0,
                    "expansion_max_height": 1
                },

                # 函数集
                "function_set": {
                    "operators": ["Add", "Sub", "Mul", "Div", "Max", "Min", "Mean", "Ln", "Log", "Squ", "Cub", "Sqrt", "Cbrt"],
                    "enable_ephemeral_constant": [0, 3000]
                },

                # 残差拟合配置
                "residual_fitting": {
                    "enable": True,
                    "models": ["CatBoost", "ExtraTrees", "RandomForest", "GradientBoosting", "LightGBM", "DNN"]
                },

                # Baseline模型配置
                "baseline": {
                    "enable": True,
                    "model_names": ["LinearRegression", "RidgeCV", "ElasticNet", "DecisionTree", "DecisionTree_2", "DecisionTree_4", "DecisionTree_6", "RandomForest", "ExtraTrees", "AdaBoost", "GradientBoosting", "XGBoost", "LightGBM", "MLP", "SVR", "KNeighbors", "CatBoost", "DNN"]
                },

                # SHAP配置
                "shap": {
                    "open": False,
                    "background_sample_size": None,  # None 表示使用全部训练集
                    "explain_sample_size": None      # None 表示使用全部测试集
                }
            }

    # 4. 配置名称与 GP 参数字典（单套配置：整份 dict 即 gp_config_dict）
    config_name = "配置"
    gp_config_dict = gp_config_item
    
    # 5. 构建GP配置
    gp_config_builder = GPConfigBuilder(gp_config_dict)
    gp_config = gp_config_builder.build_config()
    
    # 6. 执行GP（传递配置名称作为输出目录名称）
    gp_executor = GPExecutor(data_config_path, gp_config, random_seeds, output_dir_name=config_name)
    saved_files = gp_executor.execute()
    
    # 7. 输出结果
    print(f"GP执行完成，共生成 {len(saved_files)} 个结果文件")
    for file_path in saved_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()

