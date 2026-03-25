"""
测试响应截断问题
诊断为什么响应会被截断
"""
import sys
import os
import json

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from llm_new_feature.llm_api import call_llm_api_with_config
from llm_new_feature.llm_prompt import build_batch_feature_generation_prompt
from llm_new_feature.api_defaults_loader import get_fallback_api_credentials

def test_api_response():
    """测试API响应，检查截断原因"""
    print("=" * 80)
    print("测试响应截断问题")
    print("=" * 80)
    
    # 测试配置
    target_name = "Ash_Deformation"
    feature_names = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O"]
    num_features = 5
    max_depth = 2
    
    _c = get_fallback_api_credentials()
    # API配置 - 使用gemini-3-pro-all
    api_config = {
        'api_key': _c['api_key'],
        'api_base_url': _c['api_base_url'],
        'model': _c['model'] or 'gemini-3-pro-all',
        'timeout': 500,
        'temperature': 0.8,
        'max_tokens': None,  # 不限制
        'max_retries': 1,
        'retry_delay': 1
    }
    if not api_config['api_key'] or not api_config['api_base_url']:
        print("请先在 utils/llm_new_feature/streamlit_prev/api_defaults.json 中填写 api_key 与 api_base_url。")
        return
    
    print(f"\n1. API配置")
    print("-" * 80)
    print(f"模型: {api_config['model']}")
    print(f"max_tokens: {api_config['max_tokens']} (None表示不限制)")
    print(f"API URL: {api_config['api_base_url']}")
    
    # 构建prompt
    print(f"\n2. 构建Prompt")
    print("-" * 80)
    prompt = build_batch_feature_generation_prompt(
        target_name, feature_names, num_features, None, None, max_depth
    )
    print(f"Prompt长度: {len(prompt)} 字符")
    print(f"Prompt前200字符: {prompt[:200]}...")
    
    # 直接调用API，获取完整响应
    print(f"\n3. 调用API并检查响应")
    print("-" * 80)
    
    import requests
    
    url = f"{api_config['api_base_url']}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config['api_key']}"
    }
    
    payload = {
        "model": api_config['model'],
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": api_config['temperature']
    }
    
    # 不添加max_tokens（让API使用默认值）
    print(f"请求payload（不包含max_tokens）: {json.dumps(payload, ensure_ascii=False, indent=2)[:300]}...")
    
    try:
        print("\n发送API请求...")
        response = requests.post(url, headers=headers, json=payload, timeout=500)
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n4. 检查API响应结构")
            print("-" * 80)
            print(f"响应键: {list(result.keys())}")
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                
                print(f"\n5. 检查choice结构")
                print("-" * 80)
                print(f"Choice键: {list(choice.keys())}")
                
                # 检查finish_reason
                finish_reason = choice.get("finish_reason", "未找到")
                print(f"\nfinish_reason: {finish_reason}")
                if finish_reason == "length":
                    print("⚠️ 响应被截断！原因是finish_reason=length")
                    print("   这意味着达到了max_tokens限制")
                elif finish_reason == "stop":
                    print("✅ 响应正常完成（finish_reason=stop）")
                else:
                    print(f"ℹ️ finish_reason: {finish_reason}")
                
                # 检查message结构
                message = choice.get("message", {})
                print(f"\n6. 检查message结构")
                print("-" * 80)
                print(f"Message键: {list(message.keys())}")
                
                # 检查content
                content = message.get("content", "")
                print(f"\ncontent字段:")
                print(f"  类型: {type(content)}")
                print(f"  长度: {len(content)} 字符")
                print(f"  是否为空: {not content}")
                if content:
                    print(f"  前200字符: {content[:200]}...")
                    print(f"  后200字符: ...{content[-200:]}")
                
                # 检查reasoning_content（gemini特有）
                reasoning_content = message.get("reasoning_content", "")
                print(f"\nreasoning_content字段:")
                print(f"  类型: {type(reasoning_content)}")
                print(f"  长度: {len(reasoning_content)} 字符")
                print(f"  是否为空: {not reasoning_content}")
                if reasoning_content:
                    print(f"  前200字符: {reasoning_content[:200]}...")
                
                # 检查usage信息
                if "usage" in result:
                    usage = result["usage"]
                    print(f"\n7. Token使用情况")
                    print("-" * 80)
                    print(f"Usage键: {list(usage.keys())}")
                    for key, value in usage.items():
                        print(f"  {key}: {value}")
                
                # 检查是否有max_tokens限制
                print(f"\n8. 检查响应中的限制信息")
                print("-" * 80)
                if "usage" in result:
                    usage = result["usage"]
                    completion_tokens = usage.get("completion_tokens", 0)
                    if finish_reason == "length":
                        print(f"⚠️ 生成的token数: {completion_tokens}")
                        print(f"   响应被截断，说明达到了某个token限制")
                        print(f"   可能的原因:")
                        print(f"   1. API服务端有默认的max_tokens限制")
                        print(f"   2. 模型本身有最大输出限制")
                        print(f"   3. 需要在请求中明确指定更大的max_tokens值")
                
                # 尝试使用我们代码中的函数
                print(f"\n9. 使用我们的API调用函数")
                print("-" * 80)
                api_response = call_llm_api_with_config(
                    prompt,
                    max_tokens=None,  # 不限制
                    temperature=0.8,
                    api_config=api_config
                )
                
                if api_response:
                    print(f"✅ API调用成功")
                    print(f"响应长度: {len(api_response)} 字符")
                    print(f"前500字符: {api_response[:500]}...")
                    print(f"后500字符: ...{api_response[-500:]}")
                    
                    # 检查是否被截断
                    if api_response.strip().endswith(']') or api_response.strip().endswith('}'):
                        print("✅ JSON看起来是完整的（以]或}结尾）")
                    else:
                        print("⚠️ JSON可能被截断（不以]或}结尾）")
                        print(f"最后50字符: ...{api_response[-50:]}")
                else:
                    print("❌ API调用失败")
                
            else:
                print("❌ 响应中没有choices字段")
                print(f"完整响应: {json.dumps(result, ensure_ascii=False, indent=2)[:1000]}")
        else:
            print(f"❌ API请求失败")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_api_response()

