"""
测试JSON生成功能
验证整个特征生成流程是否正常工作
"""
import sys
import os
import json
import tempfile
import shutil

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from llm_new_feature.feature_generator import generate_features, save_features_to_json
from llm_new_feature.tree_height import calculate_tree_height
from llm_new_feature.api_defaults_loader import get_fallback_api_credentials

def test_json_generation():
    """测试JSON生成功能"""
    print("=" * 80)
    print("开始测试JSON生成功能")
    print("=" * 80)
    _c = get_fallback_api_credentials()
    print(f"模型: {_c['model'] or '(未配置，将使用下方占位)'}")
    print(f"API URL: {_c['api_base_url'] or '(未配置)'}")
    print(f"API Key: {'已配置' if _c['api_key'] else '(未配置)'}")
    print("-" * 80)
    
    # 测试配置
    target_name = "Ash_Deformation"
    feature_names = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO"]
    num_features = 3  # 生成3个特征用于测试
    max_depth = 2  # 深度限制为2层
    
    # API配置（来自 api_defaults.json）
    api_config = {
        'api_key': _c['api_key'],
        'api_base_url': _c['api_base_url'],
        'model': _c['model'] or 'gemini-3-pro-all',
        'timeout': 500,
        'temperature': 0.8,
        'max_tokens': 2000,
        'max_retries': 2,  # 测试时减少重试次数
        'retry_delay': 1
    }
    if not api_config['api_key'] or not api_config['api_base_url']:
        print("请先在 utils/llm_new_feature/streamlit_prev/api_defaults.json 中填写 api_key 与 api_base_url。")
        return
    
    print(f"\n1. 测试特征生成（目标: {target_name}, 特征数: {num_features}, 深度限制: {max_depth}）")
    print("-" * 80)
    
    try:
        # 生成特征
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
        
        if not features:
            print("❌ 特征生成失败：未生成任何特征")
            return False
        
        print(f"✅ 成功生成 {len(features)} 个特征")
        
        # 检查每个特征
        print("\n2. 检查生成的特征")
        print("-" * 80)
        for i, feature in enumerate(features, 1):
            print(f"\n特征 {i}:")
            tree = feature.get('tree')
            description = feature.get('description', '')
            notation = feature.get('notation', '')
            height = feature.get('height', feature.get('depth'))  # 兼容旧字段名
            
            # 检查必需字段
            if not tree:
                print(f"  ❌ 缺少tree字段")
                return False
            if not description:
                print(f"  ⚠️  缺少description字段")
            if not notation:
                print(f"  ⚠️  缺少notation字段")
            
            # 计算并检查高度
            calculated_height = calculate_tree_height(tree)
            height = feature.get('height', feature.get('depth'))  # 兼容旧字段名
            if height is None:
                print(f"  ❌ 缺少height字段（或旧的depth字段）")
                return False
            
            if height != calculated_height:
                print(f"  ❌ 高度不匹配：JSON中的height={height}，计算出的高度={calculated_height}")
                return False
            
            max_height = max_depth  # 兼容旧参数名
            if max_height and height != max_height:
                print(f"  ❌ 高度不符合要求：height={height}，要求={max_height}")
                return False
            
            print(f"  ✅ tree: {json.dumps(tree, ensure_ascii=False)}")
            print(f"  ✅ description: {description[:50]}...")
            print(f"  ✅ notation: {notation}")
            print(f"  ✅ height: {height} (计算验证: {calculated_height})")
        
        # 测试JSON保存
        print("\n3. 测试JSON保存")
        print("-" * 80)
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"临时目录: {temp_dir}")
        
        try:
            filepath = save_features_to_json(
                features=features,
                target_name=target_name,
                feature_names=feature_names,
                output_dir=temp_dir
            )
            
            if not filepath:
                print("❌ JSON保存失败")
                return False
            
            print(f"✅ JSON文件已保存: {filepath}")
            
            # 验证JSON文件
            print("\n4. 验证JSON文件")
            print("-" * 80)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            # 检查顶层字段
            required_fields = ['target_name', 'feature_names', 'num_features', 'generated_at', 'features']
            for field in required_fields:
                if field not in saved_data:
                    print(f"❌ JSON文件缺少字段: {field}")
                    return False
            
            print(f"✅ 顶层字段完整")
            print(f"  - target_name: {saved_data['target_name']}")
            print(f"  - num_features: {saved_data['num_features']}")
            print(f"  - generated_at: {saved_data['generated_at']}")
            print(f"  - features数量: {len(saved_data['features'])}")
            
            # 检查feature_names格式（应该在一行）
            feature_names_str = json.dumps(saved_data['feature_names'], ensure_ascii=False)
            if '\n' in feature_names_str:
                print(f"⚠️  feature_names包含换行符（应该在一行）")
            else:
                print(f"✅ feature_names格式正确（在一行）")
            
            # 检查operands格式
            print("\n5. 检查operands格式")
            print("-" * 80)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            # 检查简单的operands数组是否在一行
            simple_operands_pattern = r'"operands":\s*\[\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
            import re
            matches = re.findall(simple_operands_pattern, json_content)
            for match in matches[:3]:  # 只检查前3个
                if '\n' in match:
                    print(f"⚠️  发现包含换行符的operands: {match[:50]}...")
                else:
                    print(f"✅ operands格式正确（在一行）: {match}")
            
            # 检查每个特征的depth字段
            print("\n6. 检查所有特征的depth字段")
            print("-" * 80)
            all_have_depth = True
            for i, feature in enumerate(saved_data['features'], 1):
                height = feature.get('height', feature.get('depth'))  # 兼容旧字段名
                if height is None:
                    print(f"❌ 特征 {i} 缺少height字段（或旧的depth字段）")
                    all_have_depth = False
                else:
                    tree = feature.get('tree')
                    if tree:
                        calculated = calculate_tree_height(tree)
                        if height != calculated:
                            print(f"⚠️  特征 {i} 高度不匹配：JSON={height}, 计算={calculated}")
                        else:
                            print(f"✅ 特征 {i} height={height}")
            
            if not all_have_depth:
                return False
            
            print("\n" + "=" * 80)
            print("✅ 所有测试通过！")
            print("=" * 80)
            print(f"\n生成的JSON文件位置: {filepath}")
            print(f"文件大小: {os.path.getsize(filepath)} 字节")
            
            # 显示JSON文件的前几行
            print("\nJSON文件预览（前20行）:")
            print("-" * 80)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]
                for line in lines:
                    print(line.rstrip())
            
            return True
            
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"\n已清理临时目录: {temp_dir}")
    
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_json_generation()
    sys.exit(0 if success else 1)

