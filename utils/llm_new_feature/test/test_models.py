"""
测试不同模型的特征生成能力
测试 gpt-3.5-turbo, gemini-3-pro-all, gpt-5
"""
import sys
import os
import json
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from llm_new_feature.feature_generator import generate_features
from llm_new_feature.api_defaults_loader import get_fallback_api_credentials

def test_model(model_name, num_features=10):
    """测试指定模型"""
    print("=" * 80)
    print(f"测试模型: {model_name}")
    print("=" * 80)
    
    # 配置
    target_name = "Ash_Deformation"
    feature_names = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O", "TiO2", "SO3"]
    max_depth = 2
    
    _c = get_fallback_api_credentials()
    if not _c['api_key'] or not _c['api_base_url']:
        print("请先在 utils/llm_new_feature/streamlit_prev/api_defaults.json 中填写 api_key 与 api_base_url。")
        return
    api_config = {
        'api_key': _c['api_key'],
        'api_base_url': _c['api_base_url'],
        'model': model_name,
        'timeout': 500,
        'temperature': 0.8,
        'max_tokens': None,  # 不限制
        'max_retries': 2,  # 测试时减少重试次数
        'retry_delay': 1
    }
    
    print(f"\n配置:")
    print(f"  目标: {target_name}")
    print(f"  特征数: {len(feature_names)}")
    print(f"  生成数量: {num_features}")
    print(f"  深度限制: {max_depth}")
    print(f"  max_tokens: {api_config['max_tokens']}")
    
    try:
        print(f"\n开始生成特征...")
        start_time = time.time()
        
        features = generate_features(
            target_name=target_name,
            feature_names=feature_names,
            num_features=num_features,
            task_context=None,
            output_dir=None,
            existing_trees=None,
            api_config=api_config,
            return_raw_response=False,
            max_depth=max_depth
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n结果:")
        print(f"  耗时: {elapsed_time:.2f} 秒")
        print(f"  生成特征数: {len(features)}/{num_features}")
        
        if features and len(features) > 0:
            print(f"  ✅ 成功生成 {len(features)} 个特征")
            
            # 显示前3个特征
            print(f"\n前3个特征示例:")
            for i, feature in enumerate(features[:3], 1):
                tree = feature.get('tree', {})
                description = feature.get('description', '')[:50]
                depth = feature.get('depth', 'N/A')
                print(f"  {i}. depth={depth}, description={description}...")
                print(f"     tree: {json.dumps(tree, ensure_ascii=False)[:100]}...")
            
            # 检查深度
            depths = [f.get('depth') for f in features if f.get('depth')]
            if depths:
                print(f"\n深度统计:")
                print(f"  所有特征深度: {depths}")
                if all(d == max_depth for d in depths):
                    print(f"  ✅ 所有特征深度都符合要求（{max_depth}）")
                else:
                    print(f"  ⚠️ 部分特征深度不符合要求")
            
            return True, len(features), features
        else:
            print(f"  ❌ 未能生成任何特征")
            return False, 0, []
            
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, []


def main():
    """主测试函数"""
    print("=" * 80)
    print("模型特征生成能力测试")
    print("=" * 80)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试配置: 生成10个特征，深度限制2层")
    print()
    
    models_to_test = [
        "gpt-3.5-turbo",
        "gemini-3-pro-all",
        "gpt-5"
    ]
    
    results = {}
    
    for model in models_to_test:
        print("\n" + "=" * 80)
        success, count, features = test_model(model, num_features=10)
        results[model] = {
            'success': success,
            'count': count,
            'features': features
        }
        print("\n" + "-" * 80)
        time.sleep(2)  # 避免请求过快
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for model, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{model:20s} {status:10s} 生成: {result['count']}/10")
    
    # 详细分析
    print("\n详细分析:")
    for model, result in results.items():
        if result['success']:
            features = result['features']
            if features:
                # 检查所有特征是否有depth字段
                has_depth = all('depth' in f for f in features)
                depths = [f.get('depth') for f in features]
                unique_depths = set(depths)
                
                print(f"\n{model}:")
                print(f"  生成特征数: {len(features)}")
                print(f"  所有特征都有depth字段: {has_depth}")
                print(f"  深度分布: {unique_depths}")
                
                # 检查是否有完整的tree结构
                has_tree = all('tree' in f and f['tree'] for f in features)
                print(f"  所有特征都有tree: {has_tree}")
                
                # 检查是否有description
                has_desc = all('description' in f and f['description'] for f in features)
                print(f"  所有特征都有description: {has_desc}")


if __name__ == "__main__":
    main()

