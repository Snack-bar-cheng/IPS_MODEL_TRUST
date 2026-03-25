# Dataset_split

**用途**：把 CSV 划成训练/测试集，对特征做 KS 检验；在输出目录生成 KS 验证图，并把路径写入 `data_config_*.json`。

## 目录布局

```
Dataset_split/
├── main.py
├── requirements.txt
├── ash_fusion_dataset_cleaned.csv   # 可换为你的数据
├── md/
│   └── JSON_STRUCTURE.md            # 字段详解（可选读）
├── utils/
│   ├── dataset_splitter.py
│   ├── generate_ks_plots.py
│   ├── ks_test.py
│   └── json_utils.py
└── dataset_onfiguration/
    └── {随机种子}_{时间戳}/
        ├── trainset_{随机种子}_{时间戳}.csv 【用于最终训练】
        ├── testset_{随机种子}_{时间戳}.csv 【用于最终测试】
        ├── data_config_{随机种子}_{时间戳}.json
        └── ks_plots/
            ├── ks_test_all_features.pdf
            └── ks_test_summary.csv
```

## 怎么用

```bash
cd Dataset_split
pip install -r requirements.txt
```

在 `main.py` 里按需求改 `dataset_path`、`target_columns`、`delete_columns`、`train_ratio`、`random_seed`，然后在**本目录**执行：

```bash
python main.py
```
## `data_config_*.json` 示例
```json
{
  "start_time": "2026-03-24 11:00:27",
  "end_time": "2026-03-24 11:00:27",
  "all_dataset_path": "C:\\Users\\user\\Dataset_split\\ash_fusion_dataset_cleaned.csv",
  "all_dataset_columns": ["Sample_ID", "Ash_Deformation", "Ash_Softening", "Ash_Fluid", "SiO2", "..."],
  "random_seed": 43,
  "set_ratio": 0.8,
  "min_p_value": 0.2704143979808486,
  "avg_p_value": 0.6715214429366049,
  "selected_feature": ["SiO2", "Al2O3", "Fe2O3", "..."],
  "target_column": ["Ash_Deformation", "Ash_Softening", "Ash_Fluid"],
  "train_set_path": "C:\\Users\\user\\Dataset_split\\dataset_onfiguration\\43_20260324_110027\\trainset_43_20260324_110027.csv",
  "test_set_path": "C:\\Users\\user\\Dataset_split\\dataset_onfiguration\\43_20260324_110027\\testset_43_20260324_110027.csv",
  "ks_plots_pdf_path": "C:\\Users\\user\\Dataset_split\\dataset_onfiguration\\43_20260324_110027\\ks_plots\\ks_test_all_features.pdf"
}
```