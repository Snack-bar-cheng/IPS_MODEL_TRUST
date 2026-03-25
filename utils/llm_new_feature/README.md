# LLM特征生成工具

## 概述

本模块将LLM特征生成与GP运行分离，提供独立的特征生成工具。

## 目录结构

```
llm_new_feature/
├── __init__.py
├── llm_api.py              # LLM API调用
├── llm_prompt.py           # Prompt构建
├── llm_parser.py           # 响应解析
├── feature_generator.py    # 特征生成主模块
├── streamlit_prev/         # Streamlit前端
│   └── app.py
└── *.json                  # 生成的JSON特征文件
```

## 使用方法

### 方法1：使用Streamlit前端（推荐）

1. 启动Streamlit应用：
```bash
cd /Users/m/Desktop/coal_project/gps/llm_new_feature/streamlit_prev
streamlit run app.py
```

2. 在浏览器中打开应用，按照界面提示：
   - 上传数据集CSV文件
   - 选择目标变量和特征列
   - 设置生成数量
   - 点击生成按钮

3. 生成的特征将保存为JSON文件在当前目录

### 方法2：使用Python API

```python
from llm_new_feature.feature_generator import generate_features

# 生成特征
filepath = generate_features(
    target_name="Ash_Deformation",
    feature_names=["SiO2", "Al2O3", "Fe2O3", ...],
    num_features=10,
    task_context="任务背景描述（可选）",
    output_dir="./"
)
```

## JSON文件格式

生成的JSON文件格式如下：

```json
{
  "target_name": "Ash_Deformation",
  "feature_names": ["SiO2", "Al2O3", ...],
  "num_features": 10,
  "generated_at": "20251128_204856",
  "features": [
    {
      "tree": {
        "operator": "Div",
        "operands": ["SiO2", "Al2O3"]
      },
      "description": "硅铝比，反映灰分中主要酸性氧化物的比例关系",
      "notation": "SiO₂ / Al₂O₃"
    },
    ...
  ]
}
```

## GP主程序集成

在GP主程序中（`main.py`），设置以下配置：

```python
llm_init_enabled = True  # 启用LLM初始化
llm_init_ratio = 0.2     # LLM特征占比20%
llm_ignore_max_depth = True  # LLM特征忽略深度限制
```

GP主程序会自动：
1. 从 `llm_new_feature/` 目录查找最新的JSON文件
2. 根据 `llm_init_ratio` 随机选择特征
3. 转换为GP格式并用于初始化种群

## 注意事项

1. 确保已安装所需依赖：`requests`, `streamlit`, `pandas`
2. LLM API配置在 `llm_api.py` 中
3. 生成的JSON文件会自动命名，包含目标变量名和时间戳
4. GP主程序会查找包含目标变量名的JSON文件

