# Genetic Programming (GP) Modeling Toolkit

## Overview

This toolkit performs symbolic regression with Genetic Programming (GP). It supports multiple initialization, selection, crossover, and mutation strategies, as well as advanced features such as LLM-guided initialization, dynamic branch expansion with High primitives, and residual fitting.

## Repository layout

```
.
├── main.py                    # Entry point
├── gp_config_builder.py       # Builds the GP configuration object
├── executor/                  # Execution layer
│   ├── gp_executor.py         # GP orchestration
│   ├── gp_evolution.py        # Evolution loop
│   ├── gp_system_setup.py     # Primitives, toolbox, operators
│   └── gp_data_loader.py      # Data loading
├── strategies/                # Strategy definitions
│   ├── initialization/
│   ├── selection/
│   ├── crossover/
│   ├── mutation/
│   ├── function_set/
│   └── high_function/
├── Baseline_models/           # Baseline regressors (used by main pipeline)
├── Dataset_split/             # Dataset split + KS diagnostics + data_config
└── utils/                     # Utilities
    ├── gp_utils/
    ├── evolution_data_saver/
    ├── evolution_visualization/
    ├── feature_importance/
    └── llm_new_feature/
```

## Quick start

### 1. Environment

```bash
pip install -r requirements.txt
```

Install PyTorch and optional Graphviz bindings as described in `requirements.txt` if you use DNN baselines/residuals or PDF tree rendering.

### 2. Configure

Edit `main.py`:

- `data_config_path` — JSON produced under `Dataset_split/dataset_onfiguration/...`
- `gp_config_dict` — population size, generations, operators, `high_function`, `residual_fitting`, `baseline`, `shap`, etc.
- `random_seeds` — one full run per seed × target column

### 3. Run

From the repository root:

```bash
python main.py
```

If `output_dir_name` is set in code (e.g. `配置`), results are written under that folder next to the project root; otherwise a timestamped `gps_result_*` directory is used.

## Outputs

For each target column, the run creates a dedicated folder containing:

- **Evolution JSON:** `evolution_process/{random_seed}_{target_name}.json`
- **Dataset info JSON:** `evolution_process/dataset_info_{target_name}.json`
- **Best-tree visualization:** `best_tree_visualization/{random_seed}_{target_name}.pdf`
- **Evolution log:** `evo_result.txt`

### Folder naming

`{output_root}/{target_column}/` (e.g. `配置/Ash_Deformation/` when using a fixed output name).

## Key features

- **Initialization:** Random and/or LLM-derived seeds (via JSON feature files).
- **Dynamic expansion:** Add High-function branches during evolution when enabled.
- **Residual fitting:** Stacking residual models (e.g. forest, boosting, DNN) on top of the symbolic model.
- **Rich logging:** Per-generation statistics and artifacts.
- **Tree visualization:** Exports the best individual’s structure (Graphviz/pygraphviz when available).

## LLM feature generation (Streamlit)

We ship a **browser-based LLM feature studio** so you can draft symbolic feature trees (JSON) separately from the GP run. The UI lives under `utils/llm_new_feature/streamlit_prev/` and talks to your configured LLM to propose features with descriptions and structured tree operands—ready to feed GP initialization via `llm_feature_path` in `main.py`.

**Prerequisites:** `pip install -r requirements.txt` (includes Streamlit and HTTP client deps). Configure API keys or endpoints in `utils/llm_new_feature/llm_api.py` and/or `streamlit_prev/api_defaults.json` as needed.

**Start the app** (from repository root):

```bash
cd utils/llm_new_feature/streamlit_prev
streamlit run app.py
```

Streamlit will print a local URL (e.g. `http://localhost:8501`). Upload your CSV, pick targets and feature columns, set how many candidates to generate, then export JSON—typically under `utils/llm_new_feature/json_save/`. Point `main.py` at those files when `initialization.llm` is enabled.

More detail: `utils/llm_new_feature/README.md`.

## Dependencies (summary)

Core stack includes **DEAP**, **NumPy**, **pandas**, **scikit-learn**, **matplotlib**, **XGBoost**, **LightGBM**, **CatBoost**, **SHAP**, and optional **PyTorch** — see `requirements.txt` for pinned versions and notes.

## Notes

1. Ensure `data_config` paths point to existing train/test CSV files (paths in JSON can be relative to the config file).
2. For LLM initialization with non-zero ratio, provide valid `llm_feature_path` JSON files.
3. Total runs = number of random seeds × number of target columns.
4. Relative and absolute paths are supported throughout.
