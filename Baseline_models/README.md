# Baseline模型评估工具

## 功能说明

该工具用于对数据集进行baseline模型评估，支持多种机器学习模型，包括线性模型、树模型、梯度提升模型等。工具会自动进行K折交叉验证，并生成详细的评估结果JSON文件。

## 文件结构

```
Baseline_models/
├── baseline_pool.py      # Baseline模型池类
├── baseline_executor.py  # Baseline执行类
├── main.py               # 主程序入口
├── utils/                # 工具函数模块
│   ├── __init__.py
│   ├── model_creator.py  # 模型创建工具
│   ├── evaluator.py      # 模型评估工具
│   └── metrics.py        # 评估指标计算
├── README.md
└── requirements.txt
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

在 `main.py` 中修改以下参数：

```python
# 数据配置文件路径（相对路径或绝对路径）
data_config_path = "../Dataset_split/dataset_onfiguration/43_20251203_230442/data_config_43_20251203_230442.json"

# 要使用的模型名称列表
model_names = ["LinearRegression", "RidgeCV", "RandomForest", "XGBoost", "LightGBM"]

# 随机种子列表（每个种子会运行一次）
random_seeds = [43, 44, 45]

# 交叉验证折数
cv_folds = 5
```

### 3. 运行程序

```bash
cd Baseline_models
python main.py
```

## 可用模型

工具支持以下baseline模型：

- **线性模型**: LinearRegression, RidgeCV, ElasticNet
- **树模型**: DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoosting
- **梯度提升模型**: XGBoost, LightGBM, CatBoost
- **神经网络**: MLP
- **其他方法**: SVR, KNeighbors

## 输入要求

### 数据配置文件格式

数据配置文件（JSON格式）应包含以下字段：

```json
{
  "train_set_path": "训练集CSV文件路径",
  "test_set_path": "测试集CSV文件路径",
  "selected_feature": ["特征1", "特征2", ...],
  "target_column": ["目标列1", "目标列2", ...]
}
```

**注意**：
- 如果`target_column`是列表且包含多个元素，工具会自动遍历每个目标列进行预测
- 所有路径支持相对路径和绝对路径

## 输出说明

### 输出目录

程序会在 `Baseline_models/` 目录下创建结果文件夹：

- **文件夹命名**: `baseline_result_{时间戳}`
- 例如: `baseline_result_20251203_231500`

### 输出文件

每个随机种子和目标列组合会生成一个JSON文件：

- **文件命名**: `baseline_{random_seed}_{timestamp}.json`
- 例如: `baseline_43_20251203_231500.json`

### JSON文件格式

```json
{
  "baseline_info": {
    "baseline_models": [
      {
        "model_name": "LinearRegression",
        "cv_metrics": {
          "folds": [
            {"r2": [0.639, 0.645, ...]},
            {"mse": [33236.8, 36214.3, ...]},
            {"rmse": [182.3, 190.3, ...]},
            {"mae": [143.6, 149.2, ...]}
          ],
          "summary": {
            "r2_mean": 0.650,
            "r2_std": 0.008,
            "mse_mean": 33350.2,
            "mse_std": 1568.4,
            "rmse_mean": 182.57,
            "rmse_std": 4.25,
            "mae_mean": 143.24,
            "mae_std": 3.75
          }
        },
        "train_metrics": {
          "r2": 0.665,
          "mse": 32081.9,
          "rmse": 179.11,
          "mae": 141.68
        },
        "test_metrics": {
          "r2": 0.649,
          "mse": 32541.3,
          "rmse": 180.39,
          "mae": 139.46
        }
      }
    ]
  }
}
```

## 评估指标

每个模型会计算以下评估指标：

- **R² (决定系数)**: 衡量模型拟合优度
- **MSE (均方误差)**: 预测误差的平方均值
- **RMSE (均方根误差)**: MSE的平方根
- **MAE (平均绝对误差)**: 预测误差的绝对值均值

每个指标都会提供：
- **CV指标**: K折交叉验证的均值和标准差
- **训练集指标**: 在完整训练集上的表现
- **测试集指标**: 在测试集上的表现

## 注意事项

1. 确保数据配置文件路径正确
2. 确保训练集和测试集CSV文件存在且格式正确
3. 如果使用CatBoost，需要单独安装：`pip install catboost`
4. 随机种子列表的长度决定了运行次数（每个种子 × 每个目标列）
5. 所有路径支持相对路径（相对于Baseline_models目录）

## 示例

假设数据配置文件包含3个目标列，随机种子列表有3个元素，模型列表有5个模型：

- 总运行次数 = 3个目标列 × 3个随机种子 = 9次
- 每次运行 = 5个模型
- 总共会生成9个JSON文件

