"""
专门测试Gemini模型的响应格式
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

def test_gemini():
    """测试Gemini模型的响应"""
    print("=" * 80)
    print("测试Gemini模型响应格式")
    print("=" * 80)
    
    creds = get_fallback_api_credentials()
    api_key = creds["api_key"]
    api_base_url = creds["api_base_url"]
    model = creds["model"] or "gemini-3-pro-all"
    if not api_key or not api_base_url:
        print("请先在 utils/llm_new_feature/streamlit_prev/api_defaults.json 中填写 api_key 与 api_base_url。")
        return
    
    # 配置
    target_name = "Ash_Deformation"
    feature_names = ["SiO2", "Al2O3", "Fe2O3", "CaO"]
    num_features = 3  # 只生成3个用于测试
    max_depth = 2
    
    # 构建prompt
    prompt = build_batch_feature_generation_prompt(
        target_name, feature_names, num_features, None, None, max_depth
    )
    
    # API调用
    url = f"{api_base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 100000
    }
    
    print(f"\n发送请求...")
    print(f"模型: {model}")
    print(f"Prompt长度: {len(prompt)} 字符")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n响应结构:")
            print(f"响应键: {list(result.keys())}")
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})
                
                print(f"\nChoice结构:")
                print(f"Choice键: {list(choice.keys())}")
                print(f"finish_reason: {choice.get('finish_reason', 'N/A')}")
                
                print(f"\nMessage结构:")
                print(f"Message键: {list(message.keys())}")
                
                # 检查所有可能的字段
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content", "")
                
                print(f"\n内容字段:")
                print(f"content长度: {len(content)} 字符")
                print(f"reasoning_content长度: {len(reasoning_content)} 字符")
                
                # 显示实际内容
                actual_content = content or reasoning_content or ""
                if actual_content:
                    print(f"\n实际响应内容（前1000字符）:")
                    print("-" * 80)
                    print(actual_content[:1000])
                    print("-" * 80)
                    
                    print(f"\n实际响应内容（后500字符）:")
                    print("-" * 80)
                    print(actual_content[-500:])
                    print("-" * 80)
                    
                    # 检查是否包含JSON
                    if '[' in actual_content and '{' in actual_content:
                        print("\n✅ 响应中包含JSON结构")
                        # 尝试提取JSON
                        import re
                        array_match = re.search(r'\[[\s\S]*\]', actual_content, re.DOTALL)
                        if array_match:
                            json_str = array_match.group(0)
                            print(f"提取的JSON长度: {len(json_str)} 字符")
                            try:
                                parsed = json.loads(json_str)
                                print(f"✅ JSON解析成功，包含 {len(parsed)} 个元素")
                            except json.JSONDecodeError as e:
                                print(f"❌ JSON解析失败: {e}")
                                print(f"JSON前500字符: {json_str[:500]}")
                                print(f"JSON后500字符: {json_str[-500:]}")
                    else:
                        print("\n⚠️ 响应中不包含明显的JSON结构")
                        print("可能的原因:")
                        print("1. Gemini返回的是文本描述而不是JSON")
                        print("2. 需要在prompt中更明确地要求JSON格式")
                        print("3. 可能需要使用不同的prompt格式")
                else:
                    print("\n❌ 没有找到任何内容")
                    
                # 检查usage
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\nToken使用:")
                    for key, value in usage.items():
                        print(f"  {key}: {value}")
            else:
                print("\n❌ 响应中没有choices")
                print(f"完整响应: {json.dumps(result, ensure_ascii=False, indent=2)[:1000]}")
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(f"响应: {response.text[:500]}")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gemini()

