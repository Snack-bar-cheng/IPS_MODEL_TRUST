"""
LLM API调用模块
负责与大模型API的交互。API Key / Base URL / 模型名等默认值仅来自 streamlit_prev/api_defaults.json，不在代码中硬编码。
"""
import logging
import requests
import json
from typing import Optional, Dict

from .api_defaults_loader import (
    get_fallback_api_credentials,
    MAX_RETRIES,
    RETRY_DELAY,
    AVAILABLE_MODELS,
)

logger = logging.getLogger(__name__)

# 供外部 from llm_new_feature.llm_api import MAX_RETRIES 等使用（与 api_defaults_loader 一致）
__all__ = [
    "call_llm_api",
    "call_llm_api_with_config",
    "MAX_RETRIES",
    "RETRY_DELAY",
    "AVAILABLE_MODELS",
]


def call_llm_api(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> Optional[str]:
    """
    调用大模型API生成响应（使用默认配置）

    参数:
        prompt: 输入提示词
        max_tokens: 最大生成token数，如果为None则不限制（使用模型的最大值）
        temperature: 温度参数

    返回:
        模型响应文本，失败返回None
    """
    return call_llm_api_with_config(prompt, max_tokens, temperature)


def call_llm_api_with_config(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    api_config: Dict = None,
) -> Optional[str]:
    """
    调用大模型API生成响应（支持自定义配置）

    参数:
        prompt: 输入提示词
        max_tokens: 最大生成token数，如果为None则不限制（使用模型的最大值）
        temperature: 温度参数
        api_config: API配置字典，包含api_key, api_base_url, model, timeout

    返回:
        模型响应文本，失败返回None
    """
    if api_config is None:
        api_config = {}

    fb = get_fallback_api_credentials()
    api_key = (api_config.get("api_key") or "").strip() or fb["api_key"]
    api_base_url = (api_config.get("api_base_url") or "").strip() or fb["api_base_url"]
    model = (api_config.get("model") or "").strip() or fb["model"]
    timeout = api_config.get("timeout", 500)

    if not api_key or not api_base_url or not model:
        logger.error(
            "缺少 API 配置：请在 streamlit_prev/api_defaults.json 中填写 api_key、api_base_url、model，"
            "或通过 api_config 传入。"
        )
        return None

    url = f"{api_base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
    }

    # 处理max_tokens
    # 如果为None，设置一个很大的值以确保不被截断（某些API服务端可能有默认限制）
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    else:
        # 即使前端设置为None（不限制），也传递一个很大的值给API
        # 因为某些API服务端可能有默认的max_tokens限制（如8192）
        # 设置一个足够大的值（100000）以确保不会被截断
        payload["max_tokens"] = 100000
        logger.debug("max_tokens为None，设置为100000以确保不被截断")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})

                # 检查finish_reason，如果是length说明响应被截断
                finish_reason = choice.get("finish_reason", "")
                if finish_reason == "length":
                    logger.warning("⚠️ API响应被截断（finish_reason=length），可能需要增加max_tokens或设置为None（不限制）")
                    # 记录当前max_tokens设置
                    current_max_tokens = payload.get("max_tokens", "未设置（使用模型默认值）")
                    logger.warning(f"当前max_tokens设置: {current_max_tokens}")

                # 获取content内容
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content", "")

                # 对于gemini模型，可能同时有content和reasoning_content
                # 优先使用content（包含实际响应），如果content为空才使用reasoning_content
                # 注意：reasoning_content通常是推理过程，而content是实际输出
                if not content and reasoning_content:
                    content = reasoning_content
                    logger.info(f"content为空，使用reasoning_content作为响应内容（长度{len(reasoning_content)}）")
                elif content and reasoning_content:
                    logger.info(f"同时存在content（长度{len(content)}）和reasoning_content（长度{len(reasoning_content)}），优先使用content")
                elif content:
                    logger.debug(f"使用content作为响应内容（长度{len(content)}）")

                # 如果content仍然为空，尝试从其他字段获取
                if not content:
                    # 尝试从整个message中查找内容
                    for key in ["content", "text", "response"]:
                        if key in message and message[key]:
                            content = message[key]
                            logger.info(f"从字段 {key} 获取响应内容")
                            break

                if content:
                    return content
                else:
                    logger.warning(f"API响应中没有找到有效内容，完整响应: {json.dumps(result, ensure_ascii=False)[:500]}")
                    return None
        else:
            logger.warning(f"API请求失败，状态码: {response.status_code}, 响应: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        logger.error(f"API请求超时（{timeout}秒）")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"API连接错误: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求异常: {e}")
        return None
    except Exception as e:
        logger.error(f"调用大模型API时出错: {e}")
        return None
