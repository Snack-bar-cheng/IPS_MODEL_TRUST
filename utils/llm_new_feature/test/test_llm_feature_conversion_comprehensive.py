"""
全面测试LLM特征转换 - 检查所有特征并找出问题
"""
import sys
import os
import json
import logging
import itertools
import random
from inspect import isclass

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from deap import gp, base, creator
from utils.strong_gp_data_types import Float1, Vector1
from evolution_process.function_set_strategy.arithmetic_operators import *
from evolution_process.function_set_strategy.feature_combination_strategies import root_con10
from utils.llm_new_feature.llm_to_gp_converter import load_llm_features, llm_tree_to_gp_expr

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_pset(feature_names):
    """创建测试用的原语集"""
    pset = gp.PrimitiveSetTyped('MAIN', itertools.repeat(Float1, len(feature_names)), Vector1)
    
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f'ARG{i}': name})
    
    pset.addPrimitive(root_con10, [Float1,Float1,Float1,Float1,Float1,Float1,Float1,Float1,Float1,Float1], Vector1, name='High_10')
    pset.addPrimitive(addition, [Float1, Float1], Float1, name="Add")
    pset.addPrimitive(subtraction, [Float1, Float1], Float1, name="Sub")
    pset.addPrimitive(multiplication, [Float1, Float1], Float1, name="Mul")
    pset.addPrimitive(division, [Float1, Float1], Float1, name='Div')
    pset.addPrimitive(maximum, [Float1, Float1], Float1, name='Max')
    pset.addPrimitive(minimum, [Float1, Float1], Float1, name='Min')
    pset.addPrimitive(mean, [Float1, Float1], Float1, name='Mean')
    pset.addPrimitive(log_transform, [Float1], Float1, name='Ln')
    pset.addPrimitive(log10_transform, [Float1], Float1, name='Log')
    pset.addPrimitive(square, [Float1], Float1, name='Squ')
    pset.addPrimitive(cube, [Float1], Float1, name='Cub')
    pset.addPrimitive(cube_root, [Float1], Float1, name='Cbrt')
    pset.addPrimitive(square_root, [Float1], Float1, name='Sqrt')
    pset.addEphemeralConstant("RandFloat", lambda: random.uniform(0, 3000), Float1)
    
    return pset


def test_all_features(llm_features, feature_names):
    """
    测试所有特征的转换
    """
    logger.info("=" * 80)
    logger.info("全面测试：转换所有LLM特征")
    logger.info("=" * 80)
    
    pset = create_test_pset(feature_names)
    
    success_count = 0
    fail_count = 0
    errors = []
    error_details = []
    
    # 统计信息
    expr_lengths = []
    operators_used = set()
    features_used = set()
    
    for i, feature in enumerate(llm_features):
        tree_dict = feature.get("tree", {})
        description = feature.get("description", "")
        
        if not tree_dict:
            fail_count += 1
            errors.append(i)
            error_details.append({
                "index": i,
                "error": "缺少tree字段",
                "description": description[:50]
            })
            continue
        
        try:
            expr = llm_tree_to_gp_expr(tree_dict, pset, feature_names)
            
            if expr and len(expr) > 0:
                success_count += 1
                expr_lengths.append(len(expr))
                
                # 统计使用的操作符和特征
                for item in expr:
                    if hasattr(item, 'name'):
                        operators_used.add(item.name)
                    elif isinstance(item, gp.Terminal):
                        if hasattr(item, 'value'):
                            if isinstance(item.value, str):
                                features_used.add(item.value)
            else:
                fail_count += 1
                errors.append(i)
                error_details.append({
                    "index": i,
                    "error": "表达式为空",
                    "description": description[:50]
                })
        except Exception as e:
            fail_count += 1
            errors.append(i)
            error_details.append({
                "index": i,
                "error": str(e),
                "description": description[:50],
                "tree": json.dumps(tree_dict, ensure_ascii=False)[:200]
            })
    
    # 输出统计结果
    logger.info(f"\n转换结果统计:")
    logger.info(f"  总特征数: {len(llm_features)}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  成功率: {success_count/len(llm_features)*100:.2f}%")
    
    if expr_lengths:
        logger.info(f"\n表达式长度统计:")
        logger.info(f"  最小长度: {min(expr_lengths)}")
        logger.info(f"  最大长度: {max(expr_lengths)}")
        logger.info(f"  平均长度: {sum(expr_lengths)/len(expr_lengths):.2f}")
    
    logger.info(f"\n使用的操作符 ({len(operators_used)}): {sorted(operators_used)}")
    logger.info(f"使用的特征数: {len(features_used)}")
    
    if errors:
        logger.warning(f"\n发现 {len(errors)} 个失败的特征:")
        for detail in error_details[:10]:  # 只显示前10个
            logger.warning(f"  索引 {detail['index']}: {detail['error']}")
            logger.warning(f"    描述: {detail['description']}")
            if 'tree' in detail:
                logger.warning(f"    Tree: {detail['tree']}")
    
    return success_count, fail_count, errors, error_details


def check_operator_mapping(llm_features, feature_names):
    """
    检查JSON中使用的操作符是否都在pset中定义
    """
    logger.info("=" * 80)
    logger.info("检查操作符映射")
    logger.info("=" * 80)
    
    pset = create_test_pset(feature_names)
    
    # 获取pset中定义的所有操作符
    available_operators = set()
    if Float1 in pset.primitives:
        for p in pset.primitives[Float1]:
            available_operators.add(p.name)
    if Vector1 in pset.primitives:
        for p in pset.primitives[Vector1]:
            available_operators.add(p.name)
    
    logger.info(f"PSet中定义的操作符: {sorted(available_operators)}")
    
    # 从JSON中提取所有使用的操作符
    used_operators = set()
    
    def extract_operators(tree):
        if isinstance(tree, dict) and "operator" in tree:
            operator = tree["operator"]
            used_operators.add(operator)
            operands = tree.get("operands", [])
            for operand in operands:
                extract_operators(operand)
    
    for feature in llm_features:
        tree_dict = feature.get("tree", {})
        extract_operators(tree_dict)
    
    logger.info(f"JSON中使用的操作符: {sorted(used_operators)}")
    
    # 检查是否有未定义的操作符
    undefined_operators = used_operators - available_operators
    
    if undefined_operators:
        logger.error(f"❌ 发现未定义的操作符: {sorted(undefined_operators)}")
        logger.error("这些操作符在JSON中使用但在pset中未定义，会导致转换失败！")
        return False
    else:
        logger.info("✅ 所有使用的操作符都在pset中定义")
        return True


def check_feature_names_in_trees(llm_features, json_feature_names):
    """
    检查tree中使用的特征名是否都在JSON的feature_names中
    """
    logger.info("=" * 80)
    logger.info("检查特征名称匹配")
    logger.info("=" * 80)
    
    json_feature_set = set(json_feature_names)
    used_feature_names = set()
    
    def extract_features(tree):
        if isinstance(tree, str):
            used_feature_names.add(tree)
        elif isinstance(tree, dict):
            operands = tree.get("operands", [])
            for operand in operands:
                extract_features(operand)
    
    for feature in llm_features:
        tree_dict = feature.get("tree", {})
        extract_features(tree_dict)
    
    # 过滤掉可能是常量的字符串
    real_features = []
    constants = []
    for name in used_feature_names:
        try:
            float(name)
            constants.append(name)
        except ValueError:
            real_features.append(name)
    
    logger.info(f"JSON中的特征名数量: {len(json_feature_names)}")
    logger.info(f"Tree中使用的特征名数量: {len(real_features)}")
    logger.info(f"Tree中可能的常量数量: {len(constants)}")
    
    # 检查不匹配的特征名
    mismatched = set(real_features) - json_feature_set
    
    if mismatched:
        logger.error(f"❌ 发现不匹配的特征名: {sorted(mismatched)}")
        logger.error("这些特征名在tree中使用但不在JSON的feature_names中，会导致转换失败！")
        return False
    else:
        logger.info("✅ 所有使用的特征名都在JSON的feature_names中")
        return True


def main():
    """主测试函数"""
    logger.info("开始全面测试LLM特征转换功能")
    logger.info("=" * 80)
    
    # 加载特征
    target_name = "Ash_Deformation"
    llm_features = load_llm_features(target_name)
    
    if not llm_features:
        logger.error("无法继续测试：加载JSON失败")
        return
    
    logger.info(f"成功加载 {len(llm_features)} 个LLM特征")
    
    # 读取feature_names
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_save_dir = os.path.join(base_dir, 'json_save')
    import glob
    pattern = os.path.join(json_save_dir, f"llm_{target_name}_*.json")
    json_files = glob.glob(pattern)
    json_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = json_files[0]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    feature_names = data.get("feature_names", [])
    
    # 测试1: 检查操作符映射
    op_check = check_operator_mapping(llm_features, feature_names)
    
    # 测试2: 检查特征名匹配
    name_check = check_feature_names_in_trees(llm_features, feature_names)
    
    # 测试3: 测试所有特征转换
    success_count, fail_count, errors, error_details = test_all_features(llm_features, feature_names)
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    logger.info(f"操作符检查: {'✅ 通过' if op_check else '❌ 失败'}")
    logger.info(f"特征名检查: {'✅ 通过' if name_check else '❌ 失败'}")
    logger.info(f"转换测试: 成功 {success_count}/{len(llm_features)}, 失败 {fail_count}/{len(llm_features)}")
    logger.info(f"总体成功率: {success_count/len(llm_features)*100:.2f}%")
    
    if fail_count > 0:
        logger.warning(f"\n⚠️  发现 {fail_count} 个转换失败的特征")
        logger.warning("建议检查这些特征的tree结构是否正确")
    else:
        logger.info("\n✅ 所有测试通过！LLM特征转换功能正常")


if __name__ == "__main__":
    main()

