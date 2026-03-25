"""
API配置组件模块
默认展示值从 streamlit_prev/api_defaults.json 读取；若文件不存在则各字段为空，由用户自行填写。
"""
import os
import streamlit as st
from llm_new_feature.api_defaults_loader import (
    load_api_defaults,
    get_api_defaults_path,
    MAX_RETRIES,
    RETRY_DELAY,
    AVAILABLE_MODELS,
)


def _coalesce_str(data, key, fallback_when_no_file=""):
    """无文件或 JSON 中该键为 null/缺失时返回空字符串。"""
    if data is None:
        return fallback_when_no_file
    v = data.get(key)
    if v is None:
        return ""
    return str(v)


def _coalesce_number(data, key, default):
    """无文件、键缺失或为 null 时使用 default。"""
    if data is None:
        return default
    v = data.get(key)
    if v is None:
        return default
    try:
        return type(default)(v)
    except (TypeError, ValueError):
        return default


def _model_options(data):
    if data and isinstance(data.get("available_models"), list) and data["available_models"]:
        return [str(x) for x in data["available_models"] if str(x).strip()]
    return list(AVAILABLE_MODELS)


def render_api_config():
    """
    渲染API配置部分

    返回:
        api_config: API配置字典
    """
    st.subheader("4. API配置")

    has_defaults_file = os.path.isfile(get_api_defaults_path())
    raw = load_api_defaults()

    col1, col2 = st.columns(2)

    api_key = _coalesce_str(raw, "api_key")
    api_base_url = _coalesce_str(raw, "api_base_url")
    model_options = _model_options(raw)
    default_model = _coalesce_str(raw, "model")
    if not default_model and has_defaults_file:
        # 文件存在但 model 为空：下拉第一项留空
        model_options = [""] + model_options
        default_index = 0
    elif default_model in model_options:
        default_index = model_options.index(default_model)
    elif not has_defaults_file:
        # 无 json：首项为空，提示用户自选
        model_options = [""] + model_options
        default_index = 0
    else:
        # 文件中有 model 但不在 available_models 中：仍显示列表，默认第一项
        default_index = 0

    with col1:
        api_key = st.text_input(
            "API Key",
            value=api_key,
            type="password",
            help="LLM API密钥。默认来自同目录 api_defaults.json；无该文件时请自行填写。",
        )
        api_base_url = st.text_input(
            "API Base URL",
            value=api_base_url,
            help="LLM API基础URL",
        )
        model_name = st.selectbox(
            "模型名称",
            options=model_options,
            index=default_index,
            help="选择使用的模型名称。支持 gemini-3-pro-all、gpt-3.5-turbo、qwen-max、grok-4、deepseek-v3.1 等模型",
        )

    timeout = _coalesce_number(raw, "timeout", 500)
    temperature = float(_coalesce_number(raw, "temperature", 0.8))
    max_retries = int(_coalesce_number(raw, "max_retries", MAX_RETRIES))
    retry_delay = int(_coalesce_number(raw, "retry_delay", RETRY_DELAY))

    enable_token_limit = False
    max_tokens_val = None
    if has_defaults_file:
        enable_token_limit = bool(raw.get("enable_token_limit", False))
        mt = raw.get("max_tokens")
        max_tokens_val = int(mt) if mt is not None else None

    with col2:
        timeout = st.number_input(
            "API超时（秒）",
            min_value=10,
            max_value=1000,
            value=int(timeout),
            help="API请求超时时间",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(temperature),
            step=0.1,
            help="生成温度参数",
        )
        enable_token_limit = st.checkbox(
            "启用Token限制",
            value=enable_token_limit,
            help="是否限制生成的最大token数",
        )
        max_tokens = None
        if enable_token_limit:
            default_mt = max_tokens_val if max_tokens_val is not None else 2000
            max_tokens = st.number_input(
                "最大Token数",
                min_value=1,
                max_value=256000,
                value=int(default_mt),
                help="生成的最大token数，默认不限制",
            )
        max_retries = st.number_input(
            "最大重试次数",
            min_value=1,
            max_value=10,
            value=int(max_retries),
            help="API调用失败时的最大重试次数",
        )
        retry_delay = st.number_input(
            "重试延迟（秒）",
            min_value=1,
            max_value=10,
            value=int(retry_delay),
            help="重试之间的延迟时间",
        )

    api_config = {
        "api_key": api_key,
        "api_base_url": api_base_url,
        "model": model_name,
        "timeout": timeout,
        "temperature": temperature,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "max_tokens": max_tokens,
    }

    return api_config
