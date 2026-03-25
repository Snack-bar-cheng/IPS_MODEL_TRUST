# LLM Feature Generation Utilities

**中文说明：** [README.md](README.md)

## Overview

This package separates **LLM-driven feature ideation** from the GP training loop. Use it to produce structured JSON artifacts that `main.py` can load for LLM-biased initialization.

## Layout

```
llm_new_feature/
├── __init__.py
├── llm_api.py              # LLM HTTP/API calls
├── llm_prompt.py           # Prompt templates
├── llm_parser.py           # Response parsing
├── feature_generator.py    # High-level generator API
├── streamlit_prev/         # Streamlit UI
│   └── app.py
├── json_save/              # Saved feature JSON files
└── ...
```

## Usage

### Option A — Streamlit UI (recommended for interactive runs)

```bash
cd utils/llm_new_feature/streamlit_prev
pip install streamlit pandas requests
streamlit run app.py
```

In the browser:

1. Upload a dataset CSV.  
2. Choose targets and feature columns.  
3. Set how many features to synthesize.  
4. Click generate.

Artifacts are written as JSON (typically under `json_save/` or the directory configured in the app).

### Option B — Python API

```python
from llm_new_feature.feature_generator import generate_features

filepath = generate_features(
    target_name="Ash_Deformation",
    feature_names=["SiO2", "Al2O3", "Fe2O3"],
    num_features=10,
    task_context="Optional domain description",
    output_dir="./json_save",
)
```

## JSON schema (features file)

```json
{
  "target_name": "Ash_Deformation",
  "feature_names": ["SiO2", "Al2O3"],
  "num_features": 10,
  "generated_at": "20251128_204856",
  "features": [
    {
      "tree": {
        "operator": "Div",
        "operands": ["SiO2", "Al2O3"]
      },
      "description": "Silica-to-alumina ratio capturing acidic oxide balance",
      "notation": "SiO₂ / Al₂O₃"
    }
  ]
}
```

## Integrating with the GP driver

In `main.py`, point `llm_feature_path` to your JSON and tune initialization:

```python
"initialization": {
    "llm": {
        "enabled": True,
        "ratio": 0.2,
        "llm_max_tree_height": 2,
        "llm_feature_path": ["/path/to/llm_target_*.json"]
    }
}
```

The executor loads trees via `llm_to_gp_converter.load_llm_features` and blends them into the initial population according to `ratio`.

## Notes

1. Install runtime deps: `requests`, `streamlit`, `pandas`, etc. (see project `requirements.txt`).  
2. API keys / endpoints belong in `llm_api.py` or your deployment secrets.  
3. JSON filenames usually embed the target name and timestamp—keep them stable when wiring into `main.py`.  
4. When `llm.paths` are missing, the loader may fall back to globbing `json_save/llm_{target}_*.json` depending on configuration.
