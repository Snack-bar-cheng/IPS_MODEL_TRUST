"""
测试LLM特征从JSON文件读取并转换为GP格式
专门用于检查转换过程中是否存在问题
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_pset(feature_names):
    """
    创建测试用的原语集
    """
    # 创建原语集
    pset = gp.PrimitiveSetTyped('MAIN', itertools.repeat(Float1, len(feature_names)), Vector1)
    
    # 动态重命名变量名
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f'ARG{i}': name})
    
    # 添加High_10原语
    pset.addPrimitive(root_con10, [Float1,Float1,Float1,Float1,Float1,Float1,Float1,Float1,Float1,Float1], Vector1, name='High_10')

    # 基础算术运算符
    pset.addPrimitive(addition, [Float1, Float1], Float1, name="Add")
    pset.addPrimitive(subtraction, [Float1, Float1], Float1, name="Sub")
    pset.addPrimitive(multiplication, [Float1, Float1], Float1, name="Mul")
    pset.addPrimitive(division, [Float1, Float1], Float1, name='Div')

    # 选择算子
    pset.addPrimitive(maximum, [Float1, Float1], Float1, name='Max')
    pset.addPrimitive(minimum, [Float1, Float1], Float1, name='Min')
    pset.addPrimitive(mean, [Float1, Float1], Float1, name='Mean')

    # 对数变换函数
    pset.addPrimitive(log_transform, [Float1], Float1, name='Ln')
    pset.addPrimitive(log10_transform, [Float1], Float1, name='Log')

    # 幂运算函数
    pset.addPrimitive(square, [Float1], Float1, name='Squ')
    pset.addPrimitive(cube, [Float1], Float1, name='Cub')

    # 根式运算函数
    pset.addPrimitive(cube_root, [Float1], Float1, name='Cbrt')
    pset.addPrimitive(square_root, [Float1], Float1, name='Sqrt')

    pset.addEphemeralConstant("RandFloat", lambda: random.uniform(0, 3000), Float1)
    
    return pset


def test_load_json():
    """
    测试1: 加载JSON文件
    """
    logger.info("=" * 80)
    logger.info("测试1: 加载LLM特征JSON文件")
    logger.info("=" * 80)
    
    target_name = "Ash_Deformation"
    llm_features = load_llm_features(target_name)
    
    if not llm_features:
        logger.error("❌ 加载失败：未找到LLM特征")
        return None
    
    logger.info(f"✅ 成功加载 {len(llm_features)} 个LLM特征")
    
    # 检查第一个特征的结构
    if llm_features:
        first_feature = llm_features[0]
        logger.info(f"第一个特征结构: {list(first_feature.keys())}")
        if "tree" in first_feature:
            logger.info(f"第一个特征的tree类型: {type(first_feature['tree'])}")
            logger.info(f"第一个特征的tree内容: {json.dumps(first_feature['tree'], indent=2, ensure_ascii=False)[:200]}...")
    
    return llm_features


def test_tree_conversion(llm_features, feature_names):
    """
    测试2: 转换tree为GP表达式
    """
    logger.info("=" * 80)
    logger.info("测试2: 转换tree为GP表达式")
    logger.info("=" * 80)
    
    # 创建pset
    pset = create_test_pset(feature_names)
    
    # 测试前10个特征
    test_count = min(10, len(llm_features))
    success_count = 0
    fail_count = 0
    errors = []
    
    for i in range(test_count):
        feature = llm_features[i]
        tree_dict = feature.get("tree", {})
        description = feature.get("description", "")
        
        logger.info(f"\n--- 测试特征 {i+1}/{test_count} ---")
        logger.info(f"描述: {description[:50]}...")
        logger.info(f"Tree类型: {type(tree_dict)}")
        
        if not tree_dict:
            logger.warning(f"⚠️  特征 {i+1} 缺少tree字段")
            fail_count += 1
            errors.append(f"特征 {i+1}: 缺少tree字段")
            continue
        
        try:
            # 转换为GP表达式
            expr = llm_tree_to_gp_expr(tree_dict, pset, feature_names)
            
            if expr and len(expr) > 0:
                logger.info(f"✅ 转换成功，表达式长度: {len(expr)}")
                logger.debug(f"表达式前5个元素: {[str(item)[:50] for item in expr[:5]]}")
                success_count += 1
            else:
                logger.error(f"❌ 转换失败：表达式为空")
                fail_count += 1
                errors.append(f"特征 {i+1}: 表达式为空")
        except Exception as e:
            logger.error(f"❌ 转换失败：{e}")
            import traceback
            logger.error(traceback.format_exc())
            fail_count += 1
            errors.append(f"特征 {i+1}: {str(e)}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"转换测试结果: 成功 {success_count}/{test_count}, 失败 {fail_count}/{test_count}")
    logger.info("=" * 80)
    
    if errors:
        logger.error("错误详情:")
        for error in errors:
            logger.error(f"  - {error}")
    
    return success_count, fail_count, errors


def test_feature_names_mismatch(llm_features):
    """
    测试4: 检查JSON中的feature_names与tree中使用的特征名是否匹配
    """
    logger.info("=" * 80)
    logger.info("测试4: 检查特征名称匹配")
    logger.info("=" * 80)
    
    # 从JSON文件读取feature_names
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_save_dir = os.path.join(base_dir, 'json_save')
    target_name = "Ash_Deformation"
    
    pattern = os.path.join(json_save_dir, f"llm_{target_name}_*.json")
    import glob
    json_files = glob.glob(pattern)
    
    if not json_files:
        logger.error("未找到JSON文件")
        return
    
    json_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = json_files[0]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_feature_names = data.get("feature_names", [])
    logger.info(f"JSON中的特征名数量: {len(json_feature_names)}")
    logger.info(f"JSON中的特征名: {json_feature_names[:10]}...")
    
    # 从tree中提取所有使用的特征名
    used_feature_names = set()
    
    def extract_features_from_tree(tree):
        """递归提取tree中使用的所有特征名"""
        if isinstance(tree, str):
            # 可能是特征名或常量
            used_feature_names.add(tree)
        elif isinstance(tree, dict):
            if "operator" in tree:
                operands = tree.get("operands", [])
                for operand in operands:
                    extract_features_from_tree(operand)
    
    # 检查前100个特征
    check_count = min(100, len(llm_features))
    for i in range(check_count):
        feature = llm_features[i]
        tree_dict = feature.get("tree", {})
        extract_features_from_tree(tree_dict)
    
    logger.info(f"从tree中提取的特征名数量: {len(used_feature_names)}")
    logger.info(f"从tree中提取的特征名: {sorted(list(used_feature_names))[:20]}")
    
    # 检查是否有不匹配的特征名
    json_feature_set = set(json_feature_names)
    mismatched = used_feature_names - json_feature_set
    
    # 过滤掉可能是常量的字符串（尝试转换为浮点数）
    real_mismatched = []
    for name in mismatched:
        try:
            float(name)  # 如果能转换为浮点数，说明是常量，不是特征名
        except ValueError:
            real_mismatched.append(name)
    
    if real_mismatched:
        logger.warning(f"⚠️  发现不匹配的特征名: {real_mismatched}")
    else:
        logger.info("✅ 所有使用的特征名都在JSON的feature_names中")


def main():
    """
    主测试函数
    """
    logger.info("开始测试LLM特征转换功能")
    logger.info("=" * 80)
    
    # 测试1: 加载JSON
    llm_features = test_load_json()
    if not llm_features:
        logger.error("无法继续测试：加载JSON失败")
        return
    
    # 从JSON文件读取feature_names
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_save_dir = os.path.join(base_dir, 'json_save')
    target_name = "Ash_Deformation"
    
    import glob
    pattern = os.path.join(json_save_dir, f"llm_{target_name}_*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        logger.error("未找到JSON文件")
        return
    
    json_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = json_files[0]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    feature_names = data.get("feature_names", [])
    logger.info(f"从JSON读取的特征名: {feature_names}")
    
    # 测试4: 检查特征名匹配
    test_feature_names_mismatch(llm_features)
    
    # 测试2: 转换tree
    success_count, fail_count, errors = test_tree_conversion(llm_features, feature_names)
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    logger.info(f"加载特征数: {len(llm_features)}")
    logger.info(f"转换测试: 成功 {success_count}, 失败 {fail_count}")
    
    if errors:
        logger.error(f"发现 {len(errors)} 个错误，详情见上方日志")
    else:
        logger.info("✅ 所有测试通过！")


if __name__ == "__main__":
    main()

