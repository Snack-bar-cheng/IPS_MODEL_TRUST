# 遗传规划（GP）模型工具

## 功能说明

该工具用于使用遗传规划（Genetic Programming, GP）算法进行符号回归，支持多种初始化策略、选择策略、交叉策略和变异策略。工具支持LLM引导的初始化、动态分支扩展、残差拟合等高级功能。

## 文件结构

```
Gps/
├── main.py                    # 主程序入口
├── gp_config_builder.py       # GP配置构建器
├── executor/                  # 执行器模块
│   ├── gp_executor.py         # GP执行器
│   ├── gp_evolution.py        # 进化循环
│   ├── gp_system_setup.py     # 系统设置
│   └── gp_data_loader.py      # 数据加载
├── strategies/                # 策略模块
│   ├── initialization/        # 初始化策略
│   ├── selection/             # 选择策略
│   ├── crossover/             # 交叉策略
│   ├── mutation/              # 变异策略
│   ├── function_set/          # 函数集策略
│   └── high_function/         # High函数策略
└── utils/                     # 工具模块
    ├── gp_utils/              # GP工具函数
    ├── evolution_data_saver/  # 进化数据保存
    └── evolution_visualization/ # 可视化工具
```

## 使用方法

### 1. 配置参数

在 `main.py` 中修改以下参数：

```python
# 数据配置文件路径
data_config_path = "path/to/data_config.json"

# GP配置
gp_config_dict = {
    "population_size": 100,
    "generations": 50,
    "initialization": {...},
    "function_set": {...},
    "high_function": {...},
    ...
}

# 随机种子列表
random_seeds = [1, 2, 3]
```

### 2. 运行程序

```bash
cd Gps
python main.py
```

## 输出说明

程序会在 `gps_result_{时间戳}/` 目录下为每个目标列创建独立的文件夹，每个文件夹包含：

- **进化过程JSON**: `evolution_process/{random_seed}_{target_name}.json`
- **数据集信息JSON**: `evolution_process/dataset_info_{target_name}.json`
- **最佳树可视化**: `best_tree_visualization/{random_seed}_{target_name}.pdf`
- **进化结果文本**: `evo_result.txt`

### 文件夹命名规则

`gps_result_{时间戳}/{目标列名}/`

例如：`gps_result_20251204_144946/Ash_Deformation/`

## 主要功能

- **多种初始化策略**: 支持随机初始化和LLM引导初始化
- **动态分支扩展**: 在进化过程中动态添加High函数分支
- **残差拟合**: 使用RandomForest或RidgeCV拟合残差，提升模型性能
- **完整结果保存**: 保存每一代的详细进化信息
- **树可视化**: 自动生成最佳个体的树结构可视化

## 依赖库

- deap
- numpy
- pandas
- scikit-learn
- matplotlib
- xgboost
- lightgbm

## 注意事项

1. 确保数据配置文件路径正确
2. 如果使用LLM初始化，需要提供LLM特征文件路径
3. 随机种子列表的长度决定了运行次数（每个种子 × 每个目标列）
4. 所有路径支持相对路径和绝对路径

