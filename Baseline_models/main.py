"""
主程序入口
"""

import warnings
import numpy as np
# 过滤 sklearn 和 numpy 的 RuntimeWarning 警告（溢出、无效值、除以零等）
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
# 设置 numpy 的错误处理模式为忽略
np.seterr(all='ignore')

from baseline_pool import Baseline_Pool
from baseline_executor import Baseline_Executor

# 配置参数
data_config_path = "/Users/m/Desktop/new_sci_ash_aft/ash_aft_all/Dataset_split/dataset_onfiguration/43_20251203_235329/data_config_43_20251203_235329.json"
# model_names = ["LinearRegression", "RidgeCV", "ElasticNet", "DecisionTree", "RandomForest", "ExtraTrees", "AdaBoost", "GradientBoosting", "XGBoost", "LightGBM", "MLP", "SVR", "KNeighbors", "CatBoost", "DNN"]
# 默认示例：包含传统模型与DNN（DNN 会自动选择 GPU/MPS/CPU）
model_names = ["LinearRegression", "ExtraTrees", "DNN"]
random_seeds = list(range(1, 2))  # 1-30
cv_folds = 5

# 创建Baseline模型池
baseline_pool = Baseline_Pool(model_names=model_names)
model_config = baseline_pool.get_config_json()

# 创建Baseline执行器并执行
executor = Baseline_Executor(
    data_config_path=data_config_path,
    random_seeds=random_seeds,
    model_config=model_config,
    cv_folds=cv_folds
)

saved_files = executor.execute()

print(f"\n所有结果已保存，共 {len(saved_files)} 个文件")

