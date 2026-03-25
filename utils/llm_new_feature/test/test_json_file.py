"""
测试LLM特征JSON文件的完整性和正确性
"""
import sys
import os
import json
import logging
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from utils.llm_new_feature.llm_to_gp_converter import load_llm_features, llm_tree_to_gp_expr
from deap import gp
import itertools
from utils.strong_gp_data_types import Float1, Vector1
from evolution_process.function_set_strategy.arithmetic_operators import *
from evolution_process.function_set_strategy.feature_combination_strategies import root_con10
import random

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


def check_json_structure(file_path: str) -> Dict[str, Any]:
    """
    检查JSON文件的基本结构
    """
    logger.info("=" * 80)
    logger.info("检查1: JSON文件基本结构")
    logger.info("=" * 80)
    
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("✅ JSON文件格式正确，可以成功解析")
        
        # 检查必需字段
        required_fields = ["target_name", "feature_names", "num_features", "generated_at", "features"]
        for field in required_fields:
            if field not in data:
                issues.append(f"缺少必需字段: {field}")
                logger.error(f"❌ 缺少必需字段: {field}")
            else:
                logger.info(f"✅ 字段 '{field}' 存在")
        
        # 检查features字段
        if "features" in data:
            if not isinstance(data["features"], list):
                issues.append("features字段不是列表类型")
                logger.error(f"❌ features字段类型错误: {type(data['features'])}")
            else:
                logger.info(f"✅ features是列表，包含 {len(data['features'])} 个特征")
                
                # 检查数量是否匹配
                if "num_features" in data:
                    if len(data["features"]) != data["num_features"]:
                        issues.append(f"特征数量不匹配: 声明{data['num_features']}个，实际{len(data['features'])}个")
                        logger.warning(f"⚠️  特征数量不匹配: 声明{data['num_features']}个，实际{len(data['features'])}个")
                    else:
                        logger.info(f"✅ 特征数量匹配: {data['num_features']}个")
        
        # 检查feature_names
        if "feature_names" in data:
            if not isinstance(data["feature_names"], list):
                issues.append("feature_names字段不是列表类型")
                logger.error(f"❌ feature_names字段类型错误: {type(data['feature_names'])}")
            else:
                logger.info(f"✅ feature_names是列表，包含 {len(data['feature_names'])} 个特征名")
                logger.info(f"   特征名: {data['feature_names']}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "data": data
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析失败: {e}")
        return {
            "valid": False,
            "issues": [f"JSON解析错误: {e}"],
            "data": None
        }
    except Exception as e:
        logger.error(f"❌ 读取文件失败: {e}")
        return {
            "valid": False,
            "issues": [f"文件读取错误: {e}"],
            "data": None
        }


def check_feature_structure(features: List[Dict]) -> Dict[str, Any]:
    """
    检查每个特征的结构
    """
    logger.info("=" * 80)
    logger.info("检查2: 特征结构完整性")
    logger.info("=" * 80)
    
    issues = []
    stats = {
        "total": len(features),
        "valid": 0,
        "missing_tree": 0,
        "missing_description": 0,
        "missing_notation": 0,
        "invalid_tree": 0
    }
    
    for i, feature in enumerate(features):
        # 检查必需字段
        if "tree" not in feature:
            stats["missing_tree"] += 1
            issues.append(f"特征 {i+1}: 缺少tree字段")
            continue
        
        if "description" not in feature:
            stats["missing_description"] += 1
            issues.append(f"特征 {i+1}: 缺少description字段")
        
        if "notation" not in feature:
            stats["missing_notation"] += 1
            issues.append(f"特征 {i+1}: 缺少notation字段")
        
        # 检查tree结构
        tree = feature.get("tree", {})
        if not isinstance(tree, (dict, str)):
            stats["invalid_tree"] += 1
            issues.append(f"特征 {i+1}: tree类型错误，期望dict或str，实际{type(tree)}")
            continue
        
        # 如果是字典，检查是否有operator字段
        if isinstance(tree, dict):
            if "operator" not in tree:
                stats["invalid_tree"] += 1
                issues.append(f"特征 {i+1}: tree字典缺少operator字段")
            if "operands" not in tree:
                stats["invalid_tree"] += 1
                issues.append(f"特征 {i+1}: tree字典缺少operands字段")
        
        if stats["missing_tree"] == 0 and stats["invalid_tree"] == 0:
            stats["valid"] += 1
    
    logger.info(f"总特征数: {stats['total']}")
    logger.info(f"有效特征: {stats['valid']}")
    logger.info(f"缺少tree: {stats['missing_tree']}")
    logger.info(f"缺少description: {stats['missing_description']}")
    logger.info(f"缺少notation: {stats['missing_notation']}")
    logger.info(f"无效tree: {stats['invalid_tree']}")
    
    if issues:
        logger.warning(f"发现 {len(issues)} 个结构问题（仅显示前10个）:")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")
    
    return {
        "valid": stats["invalid_tree"] == 0 and stats["missing_tree"] == 0,
        "stats": stats,
        "issues": issues
    }


def check_tree_consistency(features: List[Dict], feature_names: List[str]) -> Dict[str, Any]:
    """
    检查tree中使用的特征名和操作符是否一致
    """
    logger.info("=" * 80)
    logger.info("检查3: Tree结构一致性")
    logger.info("=" * 80)
    
    issues = []
    used_operators = set()
    used_feature_names = set()
    invalid_operators = []
    invalid_features = []
    
    def extract_from_tree(tree: Any, path: str = ""):
        """递归提取tree中的操作符和特征名"""
        if isinstance(tree, str):
            # 可能是特征名或常量
            if tree in feature_names:
                used_feature_names.add(tree)
            else:
                # 尝试转换为数字，如果不是数字则可能是无效特征名
                try:
                    float(tree)  # 是常量
                except ValueError:
                    if path:
                        invalid_features.append(f"{path}: '{tree}'")
                    else:
                        invalid_features.append(f"'{tree}'")
        elif isinstance(tree, dict):
            operator = tree.get("operator", "")
            operands = tree.get("operands", [])
            
            if operator:
                used_operators.add(operator)
            
            for idx, operand in enumerate(operands):
                new_path = f"{path}.operands[{idx}]" if path else f"operands[{idx}]"
                extract_from_tree(operand, new_path)
    
    for i, feature in enumerate(features):
        tree = feature.get("tree", {})
        extract_from_tree(tree, f"特征{i+1}")
    
    logger.info(f"使用的操作符 ({len(used_operators)}): {sorted(used_operators)}")
    logger.info(f"使用的特征名 ({len(used_feature_names)}): {sorted(used_feature_names)}")
    
    # 检查无效特征名
    if invalid_features:
        logger.warning(f"发现 {len(invalid_features)} 个可能的无效特征名/常量（仅显示前10个）:")
        for feat in invalid_features[:10]:
            logger.warning(f"  - {feat}")
        issues.extend(invalid_features)
    
    # 检查特征名是否都在feature_names中
    invalid_feature_names = used_feature_names - set(feature_names)
    if invalid_feature_names:
        logger.error(f"❌ 发现 {len(invalid_feature_names)} 个不在feature_names中的特征名:")
        for name in sorted(invalid_feature_names):
            logger.error(f"  - '{name}'")
        issues.extend([f"无效特征名: {name}" for name in invalid_feature_names])
    
    return {
        "valid": len(invalid_feature_names) == 0 and len(invalid_features) == 0,
        "used_operators": used_operators,
        "used_feature_names": used_feature_names,
        "issues": issues
    }


def check_conversion_success(features: List[Dict], feature_names: List[str]) -> Dict[str, Any]:
    """
    检查所有特征是否能成功转换为GP表达式
    """
    logger.info("=" * 80)
    logger.info("检查4: 转换为GP表达式")
    logger.info("=" * 80)
    
    pset = create_test_pset(feature_names)
    
    success_count = 0
    fail_count = 0
    errors = []
    error_details = []
    
    for i, feature in enumerate(features):
        tree = feature.get("tree", {})
        if not tree:
            fail_count += 1
            errors.append(i)
            error_details.append({
                "index": i,
                "error": "缺少tree字段"
            })
            continue
        
        try:
            expr = llm_tree_to_gp_expr(tree, pset, feature_names)
            if expr and len(expr) > 0:
                success_count += 1
            else:
                fail_count += 1
                errors.append(i)
                error_details.append({
                    "index": i,
                    "error": "表达式为空"
                })
        except Exception as e:
            fail_count += 1
            errors.append(i)
            error_details.append({
                "index": i,
                "error": str(e),
                "tree": json.dumps(tree, ensure_ascii=False)[:200] if isinstance(tree, dict) else str(tree)[:200]
            })
    
    logger.info(f"转换结果:")
    logger.info(f"  成功: {success_count}/{len(features)}")
    logger.info(f"  失败: {fail_count}/{len(features)}")
    logger.info(f"  成功率: {success_count/len(features)*100:.2f}%")
    
    if errors:
        logger.warning(f"发现 {len(errors)} 个转换失败的特征（仅显示前10个）:")
        for detail in error_details[:10]:
            logger.warning(f"  索引 {detail['index']}: {detail['error']}")
            if 'tree' in detail:
                logger.warning(f"    Tree: {detail['tree']}")
    
    return {
        "valid": fail_count == 0,
        "success_count": success_count,
        "fail_count": fail_count,
        "errors": errors,
        "error_details": error_details
    }


def main():
    """主测试函数"""
    logger.info("开始测试LLM特征JSON文件")
    logger.info("=" * 80)
    
    # JSON文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_file = os.path.join(base_dir, 'json_save', 'llm_Ash_Deformation_20251129_215643.json')
    
    if not os.path.exists(json_file):
        logger.error(f"❌ JSON文件不存在: {json_file}")
        return
    
    logger.info(f"测试文件: {json_file}")
    logger.info(f"文件大小: {os.path.getsize(json_file) / 1024:.2f} KB")
    
    # 检查1: JSON结构
    structure_result = check_json_structure(json_file)
    if not structure_result["valid"]:
        logger.error("❌ JSON结构检查失败，无法继续测试")
        return
    
    data = structure_result["data"]
    features = data.get("features", [])
    feature_names = data.get("feature_names", [])
    
    # 检查2: 特征结构
    feature_result = check_feature_structure(features)
    
    # 检查3: Tree一致性
    consistency_result = check_tree_consistency(features, feature_names)
    
    # 检查4: 转换测试
    conversion_result = check_conversion_success(features, feature_names)
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    logger.info(f"JSON结构: {'✅ 通过' if structure_result['valid'] else '❌ 失败'}")
    logger.info(f"特征结构: {'✅ 通过' if feature_result['valid'] else '❌ 失败'}")
    logger.info(f"Tree一致性: {'✅ 通过' if consistency_result['valid'] else '❌ 失败'}")
    logger.info(f"转换测试: {'✅ 通过' if conversion_result['valid'] else '❌ 失败'}")
    logger.info(f"转换成功率: {conversion_result['success_count']}/{len(features)} ({conversion_result['success_count']/len(features)*100:.2f}%)")
    
    all_valid = (
        structure_result["valid"] and
        feature_result["valid"] and
        consistency_result["valid"] and
        conversion_result["valid"]
    )
    
    if all_valid:
        logger.info("\n✅ 所有检查通过！JSON文件没有问题")
    else:
        logger.warning("\n⚠️  发现一些问题，详情见上方日志")


if __name__ == "__main__":
    main()

