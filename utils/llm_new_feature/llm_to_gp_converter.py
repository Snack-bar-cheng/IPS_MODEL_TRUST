"""
LLM特征转换为GP表达式模块
负责将LLM生成的JSON格式特征转换为GP系统可用的表达式树
"""
import os
import json
import glob
import logging
from inspect import isclass
from deap import gp
from typing import List, Dict, Optional, Any

# 导入GP数据类型
# 注意：Float1和Vector1应该与gp_system_setup.py中定义的一致
# 在gp_system_setup.py中：Float1 = float, Vector1 = np.ndarray
import numpy as np
Float1 = float
Vector1 = np.ndarray

logger = logging.getLogger(__name__)


def load_llm_features(target_name: str = None, json_save_dir: str = None, json_file_paths: List[str] = None) -> List[Dict]:
    """
    加载LLM特征JSON文件
    
    参数:
        target_name: 预测目标名称（如 "Ash_Deformation"），当使用json_file_paths时可选
        json_save_dir: JSON文件保存目录，如果为None则使用默认目录（仅在未提供json_file_paths时使用）
        json_file_paths: JSON文件路径列表，如果提供则优先使用这些路径
    
    返回:
        features: LLM特征列表，每个元素包含tree、description、notation
    """
    all_features = []
    
    # 如果提供了文件路径列表，优先使用这些路径
    if json_file_paths and len(json_file_paths) > 0:
        logger.info(f"从用户指定的 {len(json_file_paths)} 个路径加载LLM特征")
        for json_path in json_file_paths:
            # 检查路径是否存在
            if not os.path.exists(json_path):
                logger.warning(f"LLM特征文件不存在，跳过: {json_path}")
                continue
            
            # 检查是否为JSON文件
            if not json_path.endswith('.json'):
                logger.warning(f"文件不是JSON格式，跳过: {json_path}")
                continue
            
            try:
                logger.info(f"加载LLM特征文件: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "features" in data and isinstance(data["features"], list):
                    features_count = len(data["features"])
                    all_features.extend(data["features"])
                    logger.info(f"成功从 {json_path} 加载 {features_count} 个LLM特征")
                else:
                    logger.warning(f"JSON文件格式错误，缺少 'features' 字段: {json_path}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON文件解析失败: {json_path}, 错误: {e}")
            except Exception as e:
                logger.error(f"加载LLM特征文件失败: {json_path}, 错误: {e}")
        
        if all_features:
            logger.info(f"总共加载 {len(all_features)} 个LLM特征（来自 {len(json_file_paths)} 个文件）")
        else:
            logger.warning("从指定路径未加载到任何LLM特征")
        return all_features
    
    # 如果没有提供路径列表，使用原来的逻辑（向后兼容）
    if json_save_dir is None:
        # 默认使用当前模块所在目录下的json_save目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_save_dir = os.path.join(base_dir, 'json_save')
    
    if not os.path.exists(json_save_dir):
        logger.warning(f"LLM特征目录不存在: {json_save_dir}")
        return []
    
    # 查找包含目标名称的JSON文件
    if target_name:
        pattern = os.path.join(json_save_dir, f"llm_{target_name}_*.json")
    else:
        pattern = os.path.join(json_save_dir, "llm_*.json")
    
    json_files = glob.glob(pattern)
    
    if not json_files:
        logger.warning(f"未找到LLM特征文件（模式: {pattern}）")
        return []
    
    # 按修改时间排序，使用最新的文件
    json_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = json_files[0]
    
    logger.info(f"加载LLM特征文件: {latest_file}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "features" in data and isinstance(data["features"], list):
            logger.info(f"成功加载 {len(data['features'])} 个LLM特征")
            return data["features"]
        else:
            logger.warning(f"JSON文件格式错误，缺少 'features' 字段")
            return []
    except Exception as e:
        logger.error(f"加载LLM特征文件失败: {e}")
        return []


def _fix_llm_tree_structure(tree_dict: Any, feature_names: List[str]) -> Any:
    """
    修复LLM生成的tree结构中的常见错误
    
    修复内容:
    1. 将特征名误用为操作符的情况修复为直接使用特征名
    2. 将多个操作数的Add操作转换为嵌套的Add操作
    
    参数:
        tree_dict: LLM特征树字典
        feature_names: 特征名称列表
    
    返回:
        修复后的tree字典
    """
    if isinstance(tree_dict, str):
        return tree_dict
    
    if not isinstance(tree_dict, dict) or "operator" not in tree_dict:
        return tree_dict
    
    operator_name = tree_dict["operator"]
    operands = tree_dict.get("operands", [])
    
    # 修复1: 如果操作符是特征名（误用情况），将其转换为直接使用特征名
    if operator_name in feature_names:
        # 这是一个特征名被误用为操作符的情况
        # 如果operands为空，直接返回特征名
        if not operands:
            return operator_name
        # 如果有operands，可能是错误的嵌套结构，尝试提取特征名
        # 这种情况比较复杂，暂时返回原结构让后续处理报错
    
    # 修复2: 如果Add操作有超过2个操作数，转换为嵌套结构
    if operator_name == "Add" and len(operands) > 2:
        # 递归修复所有操作数
        fixed_operands = [_fix_llm_tree_structure(op, feature_names) for op in operands]
        
        # 将多个操作数转换为嵌套的Add操作
        # 例如: Add(a, b, c) -> Add(Add(a, b), c)
        result = fixed_operands[0]
        for i in range(1, len(fixed_operands)):
            result = {
                "operator": "Add",
                "operands": [result, fixed_operands[i]]
            }
        return result
    
    # 递归修复操作数
    fixed_operands = [_fix_llm_tree_structure(op, feature_names) for op in operands]
    
    return {
        "operator": operator_name,
        "operands": fixed_operands
    }


def llm_tree_to_gp_expr(tree_dict: Any, pset: gp.PrimitiveSetTyped, feature_names: List[str]) -> List:
    """
    将LLM的tree字典结构转换为GP表达式树
    
    参数:
        tree_dict: LLM特征树字典，格式如 {"operator": "Add", "operands": [...]}
        pset: GP原语集
        feature_names: 特征名称列表
    
    返回:
        expr: GP表达式列表（前缀表示法），包含GP原语和Terminal对象
    """
    # 先尝试修复tree结构中的常见错误
    try:
        tree_dict = _fix_llm_tree_structure(tree_dict, feature_names)
    except Exception as e:
        logger.warning(f"修复tree结构时出错，使用原始结构: {e}")
        # 如果修复失败，继续使用原始结构
    
    if isinstance(tree_dict, str):
        # 如果是字符串，应该是特征名称
        if tree_dict in feature_names:
            # 找到特征索引
            idx = feature_names.index(tree_dict)
            
            # 【关键修复】pset.arguments存储的是字符串（特征名），不是Terminal对象
            # Terminal对象存储在pset.terminals[Float1]中，需要根据索引访问
            found_terminal = None
            try:
                # 从pset.terminals中查找Float1类型的终端列表（特征都是Float1类型）
                if Float1 in pset.terminals:
                    terminals = pset.terminals[Float1]
                    if idx < len(terminals):
                        found_terminal = terminals[idx]
                        # 如果Terminal是类（EphemeralConstant），需要实例化
                        if isclass(found_terminal):
                            found_terminal = found_terminal()
                    else:
                        logger.error(f"特征索引超出范围: idx={idx}, terminals长度={len(terminals)}")
                else:
                    # 如果Float1类型不存在，尝试第一个类型
                    term_types = list(pset.terminals.keys())
                    if term_types:
                        terminals = pset.terminals[term_types[0]]
                        if idx < len(terminals):
                            found_terminal = terminals[idx]
                            if isclass(found_terminal):
                                found_terminal = found_terminal()
                
                if found_terminal is not None:
                    # 验证是Terminal对象，而不是字符串
                    if isinstance(found_terminal, str):
                        logger.error(f"错误：找到的终端是字符串而不是Terminal对象: {found_terminal}")
                        found_terminal = None
                    elif not isinstance(found_terminal, gp.Terminal):
                        logger.error(f"错误：找到的对象不是Terminal类型: {type(found_terminal)}")
                        found_terminal = None
                
                if found_terminal:
                    return [found_terminal]
                else:
                    logger.error(f"无法找到特征终端: {tree_dict} (索引: {idx})")
                    raise ValueError(f"无法找到特征终端: {tree_dict} (索引: {idx})")
            except ValueError:
                raise  # 重新抛出ValueError
            except Exception as e:
                logger.error(f"获取特征终端时出错: {e}")
                raise ValueError(f"获取特征终端时出错: {e}")
        else:
            # 可能是常量，尝试转换为浮点数
            try:
                const_val = float(tree_dict)
                return [gp.Terminal(const_val, False, Float1)]
            except ValueError:
                logger.error(f"无法识别的特征或常量: {tree_dict}")
                raise ValueError(f"无法识别的特征或常量: {tree_dict}")
    
    if not isinstance(tree_dict, dict) or "operator" not in tree_dict:
        logger.error(f"无效的tree结构: {tree_dict}")
        raise ValueError(f"无效的tree结构: tree_dict必须是包含'operator'键的字典")
    
    operator_name = tree_dict["operator"]
    operands = tree_dict.get("operands", [])
    
    # 查找对应的原语（在所有类型中查找）
    prim = None
    # 遍历pset.primitives中的所有类型
    for ret_type, primitives_list in pset.primitives.items():
        for p in primitives_list:
            if p.name == operator_name:
                prim = p
                break
        if prim is not None:
                break
    
    if prim is None:
        # 如果操作符是特征名（误用情况），尝试修复
        if operator_name in feature_names:
            logger.warning(f"检测到特征名 '{operator_name}' 被误用为操作符，尝试修复...")
            # 如果operands为空，直接返回特征名
            if not operands:
                return llm_tree_to_gp_expr(operator_name, pset, feature_names)
            # 如果有operands，可能是错误的嵌套结构
            # 尝试将第一个操作数作为结果（假设是误用的特征名节点）
            if len(operands) == 1 and isinstance(operands[0], dict):
                # 可能是 {"operator": "Fe2O3", "operands": []} 这种情况
                # 尝试提取特征名
                nested_op = operands[0].get("operator", "")
                if nested_op in feature_names:
                    return llm_tree_to_gp_expr(nested_op, pset, feature_names)
        
        # 收集所有可用的操作符
        available_ops = []
        for ret_type, primitives_list in pset.primitives.items():
            available_ops.extend([p.name for p in primitives_list])
        logger.error(f"未找到操作符: {operator_name}，可用操作符: {available_ops}")
        raise ValueError(f"未找到操作符: {operator_name}，可用操作符: {available_ops}")
    
    # 验证操作数数量
    if prim.arity != len(operands):
        # 特殊处理：如果Add操作有超过2个操作数，自动转换为嵌套结构
        if operator_name == "Add" and len(operands) > 2:
            logger.warning(f"Add操作有 {len(operands)} 个操作数，自动转换为嵌套结构")
            # 转换为嵌套的Add操作
            nested_tree = operands[0]
            for i in range(1, len(operands)):
                nested_tree = {
                    "operator": "Add",
                    "operands": [nested_tree, operands[i]]
                }
            # 递归转换嵌套结构
            return llm_tree_to_gp_expr(nested_tree, pset, feature_names)
        
        logger.error(f"操作符 {operator_name} 需要 {prim.arity} 个操作数，但提供了 {len(operands)} 个")
        raise ValueError(f"操作符 {operator_name} 需要 {prim.arity} 个操作数，但提供了 {len(operands)} 个")
    
    # 递归处理操作数
    expr = [prim]
    for operand in operands:
        operand_expr = llm_tree_to_gp_expr(operand, pset, feature_names)
        if not operand_expr or len(operand_expr) == 0:
            logger.error(f"操作数转换失败: {operand}")
            raise ValueError(f"操作数转换失败: {operand}")
        expr.extend(operand_expr)
    
    # 验证表达式不为空
    if not expr or len(expr) == 0:
        logger.error(f"生成的表达式为空")
        raise ValueError(f"生成的表达式为空")
    
    return expr

