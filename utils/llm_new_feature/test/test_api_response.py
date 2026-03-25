"""
测试API响应，查看实际返回的内容
"""
import sys
import os
# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from llm_new_feature.llm_api import call_llm_api_with_config
from llm_new_feature.llm_prompt import build_batch_feature_generation_prompt
from llm_new_feature.api_defaults_loader import get_fallback_api_credentials

# 测试配置（密钥与 URL 仅来自 streamlit_prev/api_defaults.json）
_c = get_fallback_api_credentials()
target_name = "Ash_Deformation"
feature_names = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO"]
num_features = 2  # 只生成2个用于测试
max_depth = 2

api_config = {
    "api_key": _c["api_key"],
    "api_base_url": _c["api_base_url"],
    "model": _c["model"] or "gemini-3-pro-all",
    "timeout": 500,
    "temperature": 0.8,
    "max_tokens": 2000,
}

# 构建prompt
prompt = build_batch_feature_generation_prompt(
    target_name, feature_names, num_features, None, None, max_depth
)

if not api_config["api_key"] or not api_config["api_base_url"]:
    print("请先在 utils/llm_new_feature/streamlit_prev/api_defaults.json 中填写 api_key 与 api_base_url。")
    raise SystemExit(1)

print("=" * 80)
print("测试API响应")
print("=" * 80)
print(f"Prompt长度: {len(prompt)} 字符")
print(f"模型: {api_config['model']}")
print("-" * 80)
print("\n调用API...")

response = call_llm_api_with_config(
    prompt,
    max_tokens=2000,
    temperature=0.8,
    api_config=api_config
)

if response:
    print(f"\n✅ API调用成功")
    print(f"响应长度: {len(response)} 字符")
    print("\n" + "=" * 80)
    print("完整响应内容:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    # 尝试解析
    print("\n尝试解析JSON...")
    import json
    import re
    
    # 尝试提取JSON数组
    array_match = re.search(r'\[[\s\S]*\]', response)
    if array_match:
        json_str = array_match.group(0)
        print(f"\n提取的JSON字符串长度: {len(json_str)} 字符")
        print(f"前500字符:\n{json_str[:500]}")
        
        try:
            result = json.loads(json_str)
            print(f"\n✅ JSON解析成功！")
            print(f"解析到 {len(result)} 个特征")
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON解析失败: {e}")
            print(f"错误位置: {e.pos}")
            if e.pos < len(json_str):
                start = max(0, e.pos - 50)
                end = min(len(json_str), e.pos + 50)
                print(f"错误附近的文本:\n{json_str[start:end]}")
    else:
        print("❌ 无法从响应中提取JSON数组")
else:
    print("❌ API调用失败")

