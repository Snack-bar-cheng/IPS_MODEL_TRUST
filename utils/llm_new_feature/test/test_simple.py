"""
简单的JSON生成测试
使用模拟数据测试整个流程
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

from llm_new_feature.feature_generator import save_features_to_json
from llm_new_feature.tree_height import calculate_tree_height

def test_with_mock_data():
    """使用模拟数据测试JSON保存功能"""
    print("=" * 80)
    print("测试JSON保存功能（使用模拟数据）")
    print("=" * 80)
    
    # 创建模拟特征数据
    mock_features = [
        {
            "tree": {
                "operator": "Add",
                "operands": ["SiO2", "Al2O3"]
            },
            "description": "酸性氧化物总量。SiO2和Al2O3是煤炭灰分中主要的酸性氧化物，它们的总和反映了灰分的酸性特征。",
            "notation": "SiO₂ + Al₂O₃",
            "depth": 2
        },
        {
            "tree": {
                "operator": "Div",
                "operands": [
                    {
                        "operator": "Add",
                        "operands": ["Fe2O3", "CaO"]
                    },
                    {
                        "operator": "Add",
                        "operands": ["SiO2", "Al2O3"]
                    }
                ]
            },
            "description": "碱酸比，表示碱性氧化物与酸性氧化物的比值",
            "notation": "(Fe₂O₃ + CaO) / (SiO₂ + Al₂O₃)",
            "depth": 3
        },
        {
            "tree": {
                "operator": "Div",
                "operands": ["CaO", "SiO2"]
            },
            "description": "钙硅比，反映碱性氧化物CaO与酸性氧化物SiO2的比例",
            "notation": "CaO / SiO₂",
            "depth": 2
        }
    ]
    
    target_name = "Ash_Deformation"
    feature_names = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO"]
    
    print(f"\n1. 验证模拟数据的深度")
    print("-" * 80)
    for i, feature in enumerate(mock_features, 1):
        tree = feature.get('tree')
        expected_height = feature.get('height', feature.get('depth'))  # 兼容旧字段名
        calculated_height = calculate_tree_height(tree)
        if calculated_height == expected_height:
            print(f"✅ 特征 {i}: 高度={calculated_height} (正确)")
        else:
            print(f"❌ 特征 {i}: 期望高度={expected_height}, 计算高度={calculated_height}")
            return False
    
    print(f"\n2. 测试JSON保存")
    print("-" * 80)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    try:
        filepath = save_features_to_json(
            features=mock_features,
            target_name=target_name,
            feature_names=feature_names,
            output_dir=temp_dir
        )
        
        if not filepath:
            print("❌ JSON保存失败")
            return False
        
        print(f"✅ JSON文件已保存: {filepath}")
        
        # 验证JSON文件
        print(f"\n3. 验证JSON文件")
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
        with open(filepath, 'r', encoding='utf-8') as f:
            json_content = f.read()
        
        # 检查feature_names是否在一行
        feature_names_line = None
        for line in json_content.split('\n'):
            if '"feature_names"' in line:
                feature_names_line = line
                break
        
        if feature_names_line and '\n' not in feature_names_line.strip():
            print(f"✅ feature_names格式正确（在一行）")
            print(f"   {feature_names_line.strip()[:100]}...")
        else:
            print(f"⚠️  feature_names可能包含换行符")
        
        # 检查operands格式
        print(f"\n4. 检查operands格式")
        print("-" * 80)
        
        # 查找简单的operands数组（字符串数组）
        import re
        simple_operands_pattern = r'"operands":\s*\[\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
        matches = re.findall(simple_operands_pattern, json_content)
        if matches:
            for match in matches[:2]:  # 只显示前2个
                if '\n' not in match:
                    print(f"✅ operands格式正确（在一行）: {match}")
                else:
                    print(f"⚠️  operands包含换行符: {match[:50]}...")
        else:
            print("⚠️  未找到简单的operands数组（可能都是嵌套的）")
        
        # 检查每个特征的depth字段
        print(f"\n5. 检查所有特征的depth字段")
        print("-" * 80)
        all_have_depth = True
        for i, feature in enumerate(saved_data['features'], 1):
            if 'depth' not in feature:
                print(f"❌ 特征 {i} 缺少depth字段")
                all_have_depth = False
            else:
                depth = feature['depth']
                tree = feature.get('tree')
                if tree:
                    calculated = calculate_tree_height(tree)
                    if depth != calculated:
                        print(f"⚠️  特征 {i} 深度不匹配：JSON={depth}, 计算={calculated}")
                    else:
                        print(f"✅ 特征 {i} depth={depth}")
        
        if not all_have_depth:
            return False
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        print(f"\n生成的JSON文件位置: {filepath}")
        print(f"文件大小: {os.path.getsize(filepath)} 字节")
        
        # 显示JSON文件的前30行
        print("\nJSON文件预览（前30行）:")
        print("-" * 80)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:30]
            for i, line in enumerate(lines, 1):
                print(f"{i:3d}: {line.rstrip()}")
        
        return True
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n已清理临时目录: {temp_dir}")


if __name__ == "__main__":
    success = test_with_mock_data()
    sys.exit(0 if success else 1)

