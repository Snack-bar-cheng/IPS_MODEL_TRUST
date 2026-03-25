# Dataset_split

**中文说明：** [README.md](README.md)

**Purpose:** Split a CSV into train/test subsets, run Kolmogorov–Smirnov tests on features, export KS diagnostic plots, and write paths into `data_config_*.json` for downstream GP / baseline code.

## Layout

```
Dataset_split/
├── main.py
├── requirements.txt
├── ash_fusion_dataset_cleaned.csv   # Replace with your dataset
├── md/
│   └── JSON_STRUCTURE.md            # Field reference (optional)
├── utils/
│   ├── dataset_splitter.py
│   ├── generate_ks_plots.py
│   ├── ks_test.py
│   └── json_utils.py
└── dataset_onfiguration/
    └── {seed}_{timestamp}/
        ├── trainset_{seed}_{timestamp}.csv   # Training split
        ├── testset_{seed}_{timestamp}.csv    # Test split
        ├── data_config_{seed}_{timestamp}.json
        └── ks_plots/
            ├── ks_test_all_features.pdf
            └── ks_test_summary.csv
```

## How to run

```bash
cd Dataset_split
pip install -r requirements.txt
```

Edit `main.py` (`dataset_path`, `target_columns`, `delete_columns`, `train_ratio`, `random_seed`, …), then from **this directory**:

```bash
python main.py
```

## Example `data_config_*.json`

```json
{
  "start_time": "2026-03-24 11:00:27",
  "end_time": "2026-03-24 11:00:27",
  "all_dataset_path": "C:\\Users\\user\\Dataset_split\\ash_fusion_dataset_cleaned.csv",
  "all_dataset_columns": ["Sample_ID", "Ash_Deformation", "Ash_Softening", "Ash_Fluid", "SiO2"],
  "random_seed": 43,
  "set_ratio": 0.8,
  "min_p_value": 0.2704143979808486,
  "avg_p_value": 0.6715214429366049,
  "selected_feature": ["SiO2", "Al2O3", "Fe2O3"],
  "target_column": ["Ash_Deformation", "Ash_Softening", "Ash_Fluid"],
  "train_set_path": "C:\\Users\\user\\Dataset_split\\dataset_onfiguration\\43_20260324_110027\\trainset_43_20260324_110027.csv",
  "test_set_path": "C:\\Users\\user\\Dataset_split\\dataset_onfiguration\\43_20260324_110027\\testset_43_20260324_110027.csv",
  "ks_plots_pdf_path": "C:\\Users\\user\\Dataset_split\\dataset_onfiguration\\43_20260324_110027\\ks_plots\\ks_test_all_features.pdf"
}
```
