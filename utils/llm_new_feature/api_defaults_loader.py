"""
从 streamlit_prev/api_defaults.json 读取 API 默认配置（唯一配置源，不含代码内密钥）。
"""
import json
import os
from typing import Any, Dict, List, Optional

# 与 streamlit_prev/app.py 同级的 api_defaults.json
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_API_DEFAULTS_PATH = os.path.normpath(
    os.path.join(_PACKAGE_DIR, "streamlit_prev", "api_defaults.json")
)

# 当 JSON 缺失或某键为 null 时的非敏感回退（仅模型名列表等）
_FALLBACK_AVAILABLE_MODELS: List[str] = [
    "gemini-3-pro-all",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-5",
    "qwen-max",
    "grok-4",
    "deepseek-v3.1",
]
_FALLBACK_MAX_RETRIES = 5
_FALLBACK_RETRY_DELAY = 2


def get_api_defaults_path() -> str:
    return _API_DEFAULTS_PATH


def load_api_defaults() -> Dict[str, Any]:
    """读取 api_defaults.json；文件不存在或解析失败时返回空字典。"""
    if not os.path.isfile(_API_DEFAULTS_PATH):
        return {}
    try:
        with open(_API_DEFAULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _str_opt(d: Dict[str, Any], key: str) -> str:
    v = d.get(key)
    if v is None:
        return ""
    return str(v).strip()


def get_fallback_api_credentials() -> Dict[str, str]:
    """供 llm_api 等在未传入 api_config 时合并：仅来自 JSON，代码中无密钥。"""
    d = load_api_defaults()
    return {
        "api_key": _str_opt(d, "api_key"),
        "api_base_url": _str_opt(d, "api_base_url"),
        "model": _str_opt(d, "model"),
    }


def get_available_models() -> List[str]:
    d = load_api_defaults()
    m = d.get("available_models")
    if isinstance(m, list) and m:
        return [str(x) for x in m if str(x).strip()]
    return list(_FALLBACK_AVAILABLE_MODELS)


def get_max_retries_default() -> int:
    d = load_api_defaults()
    v = d.get("max_retries")
    if v is None:
        return _FALLBACK_MAX_RETRIES
    try:
        return int(v)
    except (TypeError, ValueError):
        return _FALLBACK_MAX_RETRIES


def get_retry_delay_default() -> int:
    d = load_api_defaults()
    v = d.get("retry_delay")
    if v is None:
        return _FALLBACK_RETRY_DELAY
    try:
        return int(v)
    except (TypeError, ValueError):
        return _FALLBACK_RETRY_DELAY


# 模块级常量：供 import llm_api 的代码使用（值来自 JSON + 非敏感回退）
AVAILABLE_MODELS = get_available_models()
MAX_RETRIES = get_max_retries_default()
RETRY_DELAY = get_retry_delay_default()
