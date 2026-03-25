# Baseline Models JSON 结构

## JSON 结构

```json
{
  "selected_feature": ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O", "TiO2", "SO3", "Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti", "TS", "GSAsh", "GSAsh_Dry", "Standard_Ash", "MnO"],
  "target_column": "Ash_Deformation",
  "train_set_path": "/Users/m/Desktop/new_sci_ash_aft/Dataset_split/dataset_onfiguration/43_20251203_235329/trainset_43_20251203_235329.csv",
  "test_set_path": "/Users/m/Desktop/new_sci_ash_aft/Dataset_split/dataset_onfiguration/43_20251203_235329/testset_43_20251203_235329.csv",
  "baseline_info": {
    "baseline_models": [
      {
        "model_name": "LinearRegression",
        "cv_metrics": {
          "folds": 5,
          "cross_validation": {
            "r2_mean": 0.655,
            "r2_std": 0.032,
            "mse_mean": 33350.2,
            "mse_std": 1568.4,
            "rmse_mean": 181.5,
            "rmse_std": 8.2,
            "mae_mean": 143.2,
            "mae_std": 4.5
          }
        },
        "train_metrics": {"r2": 0.665, "mse": 32081.9, "rmse": 179.11, "mae": 141.68},
        "test_metrics": {"r2": 0.649, "mse": 32541.3, "rmse": 180.39, "mae": 139.46}
      }
    ]
  }
}
```

## 字段类型

- `selected_feature`: array[string]
- `target_column`: string
- `train_set_path`: string
- `test_set_path`: string
- `baseline_info`: object
  - `baseline_models`: array[object]
    - `model_name`: string
    - `cv_metrics`: object
      - `folds`: integer
      - `cross_validation`: object
        - `r2_mean`: float
        - `r2_std`: float
        - `mse_mean`: float
        - `mse_std`: float
        - `rmse_mean`: float
        - `rmse_std`: float
        - `mae_mean`: float
        - `mae_std`: float
    - `train_metrics`: object
      - `r2`: float
      - `mse`: float
      - `rmse`: float
      - `mae`: float
    - `test_metrics`: object
      - `r2`: float
      - `mse`: float
      - `rmse`: float
      - `mae`: float
