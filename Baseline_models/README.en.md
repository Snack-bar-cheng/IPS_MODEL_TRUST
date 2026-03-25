# Baseline Model Evaluation Toolkit

**中文说明：** [README.md](README.md)

## Overview

This toolkit evaluates a suite of baseline machine-learning regressors on your dataset: linear models, tree-based models, gradient boosting, and more. It runs K-fold cross-validation and writes detailed JSON reports per run.

## Layout

```
Baseline_models/
├── baseline_pool.py      # Baseline model pool
├── baseline_executor.py  # Batch runner
├── main.py               # Standalone entry point
├── utils/
│   ├── __init__.py
│   ├── model_creator.py  # Model factory
│   ├── evaluator.py      # Training / CV / test evaluation
│   └── json_formatter.py # JSON helpers
├── README.md
└── requirements.txt
```

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `main.py`:

```python
# Path to the data_config JSON (relative or absolute)
data_config_path = "../Dataset_split/dataset_onfiguration/43_20251203_230442/data_config_43_20251203_230442.json"

model_names = ["LinearRegression", "RidgeCV", "RandomForest", "XGBoost", "LightGBM"]

random_seeds = [43, 44, 45]

cv_folds = 5
```

### 3. Run

```bash
cd Baseline_models
python main.py
```

## Supported models

- **Linear:** `LinearRegression`, `RidgeCV`, `ElasticNet`
- **Trees / ensembles:** `DecisionTree`, `RandomForest`, `ExtraTrees`, `AdaBoost`, `GradientBoosting`
- **Boosting:** `XGBoost`, `LightGBM`, `CatBoost`
- **Neural nets:** `MLP`, `DNN` (project-specific)
- **Others:** `SVR`, `KNeighbors`

## Input: `data_config` schema

```json
{
  "train_set_path": "path/to/train.csv",
  "test_set_path": "path/to/test.csv",
  "selected_feature": ["feature_1", "feature_2"],
  "target_column": ["target_1", "target_2"]
}
```

**Notes**

- If `target_column` is a list, each target is evaluated separately.
- Paths may be relative or absolute.

## Outputs

### Directory

Under `Baseline_models/`:

- Folder name: `baseline_result_{timestamp}`  
  Example: `baseline_result_20251203_231500`

### Files

Per random seed × target column:

- Filename: `baseline_{random_seed}_{timestamp}.json`  
  Example: `baseline_43_20251203_231500.json`

### JSON shape (illustrative)

```json
{
  "baseline_info": {
    "baseline_models": [
      {
        "model_name": "LinearRegression",
        "cv_metrics": {
          "folds": [
            {"r2": [0.639, 0.645]},
            {"mse": [33236.8, 36214.3]},
            {"rmse": [182.3, 190.3]},
            {"mae": [143.6, 149.2]}
          ],
          "summary": {
            "r2_mean": 0.65,
            "r2_std": 0.008,
            "mse_mean": 33350.2,
            "mse_std": 1568.4,
            "rmse_mean": 182.57,
            "rmse_std": 4.25,
            "mae_mean": 143.24,
            "mae_std": 3.75
          }
        },
        "train_metrics": {"r2": 0.665, "mse": 32081.9, "rmse": 179.11, "mae": 141.68},
        "test_metrics": {"r2": 0.649, "mse": 32541.3, "rmse": 180.39, "mae": 139.46}
      }
    ]
  }
}
```

## Metrics

For each model:

- **R²** — coefficient of determination  
- **MSE** — mean squared error  
- **RMSE** — root MSE  
- **MAE** — mean absolute error  

Each metric is reported for **CV** (mean ± std across folds), **full training set**, and **held-out test set**.

## Notes

1. Verify `data_config` paths before running.  
2. Train/test CSV files must exist and match the listed feature columns.  
3. Install CatBoost separately if needed: `pip install catboost`.  
4. Workload = (#targets) × (#random seeds); each run trains every model in `model_names`.  
5. Relative paths in config are resolved from the JSON file’s directory when possible.

## Example workload

3 targets × 3 seeds = **9** JSON files; each file contains results for **5** models ⇒ 5 fits per file.
