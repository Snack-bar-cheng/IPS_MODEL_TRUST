"""
检查JSON文件中的特定问题
"""
import json
import os

json_file = "/Users/m/Desktop/coal_project/gps/utils/llm_new_feature/json_save/llm_Ash_Deformation_20251129_215643.json"

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data["features"]
feature_names = data["feature_names"]

print("=" * 80)
print("检查特定问题")
print("=" * 80)

# 问题1: 特征名被误用为操作符
print("\n1. 检查特征名被误用为操作符:")
feature_name_as_operator = []
for i, feature in enumerate(features):
    tree = feature.get("tree", {})
    if isinstance(tree, dict):
        operator = tree.get("operator", "")
        if operator in feature_names:
            feature_name_as_operator.append({
                "index": i,
                "operator": operator,
                "tree": tree
            })

if feature_name_as_operator:
    print(f"   发现 {len(feature_name_as_operator)} 个问题:")
    for item in feature_name_as_operator:
        print(f"   特征 {item['index']+1}: operator='{item['operator']}'")
        print(f"     Tree: {json.dumps(item['tree'], ensure_ascii=False)[:150]}")
else:
    print("   ✅ 未发现问题")

# 问题2: Add操作数过多
print("\n2. 检查Add操作数过多 (>2个):")
add_too_many_operands = []
for i, feature in enumerate(features):
    def check_add_operands(tree, path=""):
        if isinstance(tree, dict):
            operator = tree.get("operator", "")
            operands = tree.get("operands", [])
            if operator == "Add" and len(operands) > 2:
                add_too_many_operands.append({
                    "index": i,
                    "operands_count": len(operands),
                    "path": path,
                    "tree": tree
                })
            for idx, op in enumerate(operands):
                new_path = f"{path}.operands[{idx}]" if path else f"operands[{idx}]"
                check_add_operands(op, new_path)
    
    tree = feature.get("tree", {})
    check_add_operands(tree)

if add_too_many_operands:
    print(f"   发现 {len(add_too_many_operands)} 个问题:")
    for item in add_too_many_operands[:5]:  # 只显示前5个
        print(f"   特征 {item['index']+1}: Add有 {item['operands_count']} 个操作数")
        print(f"     Tree: {json.dumps(item['tree'], ensure_ascii=False)[:200]}")
else:
    print("   ✅ 未发现问题")

# 问题3: 检查所有操作符的arity
print("\n3. 检查操作符操作数数量:")
operators_arity = {}
for i, feature in enumerate(features):
    def check_arity(tree):
        if isinstance(tree, dict):
            operator = tree.get("operator", "")
            operands = tree.get("operands", [])
            if operator and operator not in feature_names:  # 排除特征名
                if operator not in operators_arity:
                    operators_arity[operator] = []
                operators_arity[operator].append({
                    "expected": None,  # 需要从pset获取
                    "actual": len(operands),
                    "index": i
                })
            for op in operands:
                check_arity(op)
    
    tree = feature.get("tree", {})
    check_arity(tree)

print(f"   统计操作符使用情况:")
for op, arities in sorted(operators_arity.items()):
    counts = [a["actual"] for a in arities]
    unique_counts = sorted(set(counts))
    print(f"   {op}: 操作数数量 {unique_counts} (共{len(arities)}次使用)")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)

