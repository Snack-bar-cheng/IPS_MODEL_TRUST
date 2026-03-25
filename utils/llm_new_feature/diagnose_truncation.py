"""
诊断响应截断问题 - 简化版
"""
import sys
import os
import json
import requests

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from llm_new_feature.api_defaults_loader import get_fallback_api_credentials
from llm_new_feature.llm_prompt import build_batch_feature_generation_prompt

def diagnose():
    """诊断截断问题"""
    print("=" * 80)
    print("诊断响应截断问题")
    print("=" * 80)
    
    # 简单配置（仅来自 streamlit_prev/api_defaults.json）
    creds = get_fallback_api_credentials()
    api_key = creds["api_key"]
    api_base_url = creds["api_base_url"]
    model = creds["model"] or "gemini-3-pro-all"
    if not api_key or not api_base_url:
        print("错误：请在 utils/llm_new_feature/streamlit_prev/api_defaults.json 中填写 api_key 与 api_base_url。")
        return
    
    # 构建简单的prompt
    prompt = build_batch_feature_generation_prompt(
        "Ash_Deformation", 
        ["SiO2", "Al2O3", "Fe2O3", "CaO"], 
        3,  # 只生成3个，减少响应长度
        None, None, 2
    )
    
    print(f"\n1. 测试配置")
    print(f"模型: {model}")
    print(f"Prompt长度: {len(prompt)} 字符")
    
    # 测试1: 不传max_tokens
    print(f"\n2. 测试1: 不传max_tokens（让API使用默认值）")
    print("-" * 80)
    test_without_max_tokens(api_key, api_base_url, model, prompt)
    
    # 测试2: 传一个很大的max_tokens
    print(f"\n3. 测试2: 传max_tokens=100000")
    print("-" * 80)
    test_with_large_max_tokens(api_key, api_base_url, model, prompt)


def test_without_max_tokens(api_key, api_base_url, model, prompt):
    """测试不传max_tokens"""
    url = f"{api_base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8
        # 不传max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            analyze_response(result, "不传max_tokens")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应: {response.text[:300]}")
    except Exception as e:
        print(f"❌ 错误: {e}")


def test_with_large_max_tokens(api_key, api_base_url, model, prompt):
    """测试传很大的max_tokens"""
    url = f"{api_base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 100000  # 很大的值
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            analyze_response(result, "max_tokens=100000")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应: {response.text[:300]}")
    except Exception as e:
        print(f"❌ 错误: {e}")


def analyze_response(result, test_name):
    """分析响应"""
    print(f"\n【{test_name}】响应分析:")
    
    if "choices" not in result or len(result["choices"]) == 0:
        print("❌ 没有choices字段")
        return
    
    choice = result["choices"][0]
    message = choice.get("message", {})
    
    # 检查finish_reason
    finish_reason = choice.get("finish_reason", "未找到")
    print(f"finish_reason: {finish_reason}")
    
    if finish_reason == "length":
        print("⚠️ 响应被截断！finish_reason=length")
    elif finish_reason == "stop":
        print("✅ 响应正常完成")
    else:
        print(f"ℹ️ finish_reason: {finish_reason}")
    
    # 检查content
    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")
    
    print(f"\ncontent长度: {len(content)} 字符")
    print(f"reasoning_content长度: {len(reasoning_content)} 字符")
    
    # 使用实际内容
    actual_content = content or reasoning_content or ""
    
    if actual_content:
        print(f"\n实际响应内容:")
        print(f"总长度: {len(actual_content)} 字符")
        print(f"前300字符: {actual_content[:300]}...")
        print(f"后300字符: ...{actual_content[-300:]}")
        
        # 检查是否以]结尾（完整的JSON数组）
        trimmed = actual_content.strip()
        if trimmed.endswith(']'):
            print("✅ JSON数组看起来完整（以]结尾）")
        elif trimmed.endswith('}'):
            print("⚠️ 以}结尾，可能是单个对象或截断的数组")
        else:
            print("⚠️ 不以]或}结尾，可能被截断")
            print(f"最后50字符: ...{trimmed[-50:]}")
        
        # 尝试解析JSON
        try:
            # 尝试提取JSON数组
            import re
            array_match = re.search(r'\[[\s\S]*\]', actual_content, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                parsed = json.loads(json_str)
                print(f"✅ 成功解析JSON数组，包含 {len(parsed)} 个元素")
            else:
                print("⚠️ 无法找到完整的JSON数组")
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print("   响应可能被截断或格式不正确")
    
    # 检查usage
    if "usage" in result:
        usage = result["usage"]
        print(f"\nToken使用:")
        for key, value in usage.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    diagnose()

