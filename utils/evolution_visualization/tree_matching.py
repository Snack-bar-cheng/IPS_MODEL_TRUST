"""
LLM特征匹配相关的辅助函数
用于在GP树中查找和匹配LLM特征
"""

import re
import logging
from deap import gp
from typing import Set, List, Dict, Any


def _find_matching_llm_features(tree, llm_features, feature_names):
    """
    查找GP树中与LLM特征匹配的节点
    
    参数:
        tree: GP树个体
        llm_features: LLM特征列表
        feature_names: 特征名称列表
    
    返回:
        matching_nodes: 匹配的节点索引集合
    """
    matching_nodes = set()
    
    if not llm_features:
        return matching_nodes
    
    # 获取树的字符串表达式
    tree_str = str(tree)
    logger = logging.getLogger(__name__)
    logger.debug(f"查找匹配，树表达式: {tree_str[:100]}...")
    
    # 检查是否是High函数
    if tree_str.startswith('High_'):
        logger.debug("检测到High函数，开始解析参数")
        match = re.match(r'High_(\d+)\((.*)\)', tree_str)
        if match:
            num_args = int(match.group(1))
            args_str = match.group(2)
            
            # 解析参数（每个参数是一个特征表达式）
            # 正确解析嵌套的括号
            args = []
            depth = 0
            current_arg = ""
            for char in args_str:
                if char == '(':
                    depth += 1
                    current_arg += char
                elif char == ')':
                    depth -= 1
                    current_arg += char
                elif char == ',' and depth == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char
            if current_arg:
                args.append(current_arg.strip())
            
            # 获取树的节点和边（labels包含节点到标签的映射）
            nodes, edges, labels = gp.graph(tree)
            
            # 构建节点到子树的映射（简化：遍历所有节点，提取子树表达式）
            # 对于High函数，第一个参数对应第一个子树，以此类推
            # 使用递归方法找到每个子树的所有节点
            root = 0
            logger = logging.getLogger(__name__)
            logger.debug(f"开始匹配，GP表达式: {tree_str}")
            logger.debug(f"解析出 {len(args)} 个参数")
            
            if root < len(tree):
                # 获取High函数的所有直接子节点（参数）
                high_node = tree[root]
                if hasattr(high_node, 'arity') and high_node.arity > 0:
                    logger.debug(f"High函数arity: {high_node.arity}, 参数数量: {len(args)}")
                    
                    # 找到每个子树的根节点
                    subtree_roots = []
                    current_idx = 1  # 第一个子节点在索引1
                    for i in range(high_node.arity):
                        if current_idx < len(tree):
                            subtree_roots.append(current_idx)
                            # 计算下一个子树的起始位置
                            subtree_size = _get_subtree_size(tree, current_idx)
                            logger.debug(f"子树[{i}] 根节点: {current_idx}, 大小: {subtree_size}")
                            current_idx += subtree_size
                        else:
                            logger.warning(f"子树[{i}] 索引超出范围: {current_idx} >= {len(tree)}")
                    
                    # 对每个子树，提取表达式并与LLM特征比对
                    for idx, arg_expr in enumerate(args):
                        if idx < len(subtree_roots):
                            subtree_root = subtree_roots[idx]
                            # 提取子树的表达式（使用labels和feature_names获取特征名称）
                            subtree_expr = _extract_subtree_expression_with_names(tree, subtree_root, labels, feature_names)
                            logger.debug(f"参数[{idx}] 字符串: '{arg_expr}', 提取表达式: '{subtree_expr}'")
                            
                            # 与LLM特征比对
                            for llm_feature in llm_features:
                                llm_tree = llm_feature.get("tree", {})
                                llm_notation = llm_feature.get("notation", "")
                                
                                # 调试：记录比对过程
                                llm_expr = _llm_tree_to_expr_str(llm_tree, feature_names)
                                
                                if _compare_feature_expressions(subtree_expr, llm_tree, llm_notation, feature_names):
                                    # 找到匹配，标记该子树的所有节点
                                    logger.info(f"✓ 匹配成功: GP子树[{idx}] '{subtree_expr}' 匹配 LLM特征 '{llm_expr}'")
                                    subtree_nodes = _get_subtree_nodes(tree, subtree_root)
                                    matching_nodes.update(subtree_nodes)
                                    break
                        else:
                            logger.warning(f"参数索引 {idx} 超出子树根节点范围: {len(subtree_roots)}")
                else:
                    logger.warning(f"High节点没有arity属性或arity为0")
            else:
                logger.warning(f"根节点索引超出范围: {root} >= {len(tree)}")
    else:
        logger.debug(f"树表达式不是High函数格式，跳过匹配: {tree_str[:50]}...")
    
    logger.info(f"匹配完成，找到 {len(matching_nodes)} 个匹配节点")
    return matching_nodes


def _get_subtree_size(tree, root_idx):
    """
    获取以root_idx为根的子树的大小（节点数）
    """
    if root_idx >= len(tree):
        return 0
    
    node = tree[root_idx]
    if isinstance(node, gp.Terminal):
        return 1
    
    size = 1
    current_idx = root_idx + 1
    for _ in range(node.arity):
        child_size = _get_subtree_size(tree, current_idx)
        size += child_size
        current_idx += child_size
    
    return size


def _get_subtree_nodes(tree, root_idx):
    """
    获取以root_idx为根的子树的所有节点索引
    """
    nodes = set()
    
    def collect_nodes(idx):
        if idx >= len(tree):
            return
        nodes.add(idx)
        node = tree[idx]
        if not isinstance(node, gp.Terminal):
            current_idx = idx + 1
            for _ in range(node.arity):
                collect_nodes(current_idx)
                child_size = _get_subtree_size(tree, current_idx)
                current_idx += child_size
    
    collect_nodes(root_idx)
    return nodes


def _extract_subtree_expression(tree, root_idx):
    """
    提取以root_idx为根的子树的表达式字符串
    """
    if root_idx >= len(tree):
        return ""
    
    node = tree[root_idx]
    if isinstance(node, gp.Terminal):
        return str(node.value)
    
    # 构建函数表达式
    operator = node.name
    args = []
    current_idx = root_idx + 1
    
    for _ in range(node.arity):
        arg_expr = _extract_subtree_expression(tree, current_idx)
        args.append(arg_expr)
        child_size = _get_subtree_size(tree, current_idx)
        current_idx += child_size
    
    if len(args) == 0:
        return operator
    elif len(args) == 1:
        return f"{operator}({args[0]})"
    else:
        return f"{operator}({', '.join(args)})"


def _compare_feature_expressions(gp_expr_str, llm_tree_dict, llm_notation, feature_names):
    """
    比对GP特征表达式与LLM特征
    
    参数:
        gp_expr_str: GP特征表达式字符串
        llm_tree_dict: LLM特征的树字典
        llm_notation: LLM特征的notation
        feature_names: 特征名称列表
    
    返回:
        bool: 是否匹配
    """
    if not gp_expr_str or not llm_tree_dict:
        return False
    
    # 将LLM树转换为表达式字符串
    llm_expr = _llm_tree_to_expr_str(llm_tree_dict, feature_names)
    if not llm_expr:
        return False
    
    # 将GP表达式标准化（去除空格，统一大小写等）
    gp_normalized = gp_expr_str.replace(" ", "").strip()
    llm_normalized = llm_expr.replace(" ", "").strip()
    
    # 直接比对字符串
    if gp_normalized == llm_normalized:
        return True
    
    # 对于可交换操作（Add, Mul），尝试交换参数顺序后比对
    if _is_commutative_operation(gp_expr_str) and _is_commutative_operation(llm_expr):
        gp_swapped = _swap_operands(gp_expr_str)
        if gp_swapped:
            gp_swapped_normalized = gp_swapped.replace(" ", "").strip()
            if gp_swapped_normalized == llm_normalized:
                return True
    
    # 尝试规范化后比对（处理操作符名称差异）
    # GP中可能使用Add, Sub, Mul, Div等，需要转换为标准格式
    gp_canonical = _canonicalize_expression(gp_normalized, feature_names)
    llm_canonical = _canonicalize_expression(llm_normalized, feature_names)
    
    if gp_canonical == llm_canonical:
        return True
    
    return False


def _is_commutative_operation(expr_str):
    """检查表达式是否为可交换操作（Add, Mul, Max, Min, Mean）"""
    if not expr_str:
        return False
    expr_upper = expr_str.upper()
    return expr_upper.startswith("ADD(") or expr_upper.startswith("MUL(") or \
           expr_upper.startswith("MAX(") or expr_upper.startswith("MIN(") or \
           expr_upper.startswith("MEAN(")


def _swap_operands(expr_str):
    """交换可交换操作的参数顺序（仅用于Add和Mul）"""
    # 匹配 Operator(operand1, operand2) 格式
    match = re.match(r'^(Add|Mul)\(([^,]+),\s*([^)]+)\)$', expr_str.strip())
    if match:
        operator = match.group(1)
        operand1 = match.group(2).strip()
        operand2 = match.group(3).strip()
        return f"{operator}({operand2}, {operand1})"
    return None


def _canonicalize_expression(expr_str, feature_names):
    """
    规范化表达式字符串，使其便于比对
    
    参数:
        expr_str: 表达式字符串
        feature_names: 特征名称列表
    
    返回:
        规范化后的表达式字符串
    """
    # 替换操作符名称
    expr = expr_str
    # 将GP操作符名称替换为标准符号
    expr = expr.replace("Add", "+").replace("Sub", "-").replace("Mul", "*").replace("Div", "/")
    expr = expr.replace("Max", "max").replace("Min", "min").replace("Mean", "mean")
    
    # 移除多余的括号（可选，但可能影响比对准确性）
    # 这里保持原样，只做操作符替换
    
    return expr


def _llm_tree_to_expr_str(tree_dict, feature_names):
    """
    将LLM树字典转换为表达式字符串（与GP表达式格式一致）
    
    参数:
        tree_dict: LLM特征的树字典
        feature_names: 特征名称列表
    
    返回:
        表达式字符串（格式与GP表达式一致，如 "Div(SiO2, TiO2)"）
    """
    if isinstance(tree_dict, str):
        return tree_dict
    if isinstance(tree_dict, (int, float)):
        return str(tree_dict)
    if not isinstance(tree_dict, dict) or "operator" not in tree_dict:
        return str(tree_dict)
    
    operator = tree_dict["operator"]
    operands = tree_dict.get("operands", [])
    
    if len(operands) == 0:
        return operator
    elif len(operands) == 1:
        operand_str = _llm_tree_to_expr_str(operands[0], feature_names)
        return f"{operator}({operand_str})"
    else:
        operand_strs = [_llm_tree_to_expr_str(op, feature_names) for op in operands]
        # 使用与GP表达式一致的格式：Operator(operand1, operand2, ...)
        return f"{operator}({', '.join(operand_strs)})"


def _extract_subtree_expression_with_names(tree, root_idx, labels, feature_names):
    """
    提取以root_idx为根的子树的表达式字符串，将终端节点索引转换为特征名称
    
    参数:
        tree: GP树个体
        root_idx: 子树根节点索引
        labels: 节点到标签的映射（从gp.graph获取）
        feature_names: 特征名称列表
    
    返回:
        表达式字符串
    """
    if root_idx >= len(tree):
        return ""
    
    node = tree[root_idx]
    label = labels.get(root_idx, str(tree[root_idx]))
    
    if isinstance(node, gp.Terminal):
        # 终端节点：需要将IN16这样的标签转换为特征名称
        # 如果label是IN16格式，提取数字并映射到特征名称
        match = re.match(r'IN(\d+)', str(label))
        if match:
            idx = int(match.group(1))
            if idx < len(feature_names):
                return feature_names[idx]
        # 如果不是IN格式，可能是常量或其他，直接返回
        return str(label)
    
    # 构建函数表达式
    operator = label  # 使用label（操作符名称）
    args = []
    current_idx = root_idx + 1
    
    for _ in range(node.arity):
        arg_expr = _extract_subtree_expression_with_names(tree, current_idx, labels, feature_names)
        args.append(arg_expr)
        child_size = _get_subtree_size(tree, current_idx)
        current_idx += child_size
    
    if len(args) == 0:
        return operator
    elif len(args) == 1:
        return f"{operator}({args[0]})"
    else:
        return f"{operator}({', '.join(args)})"

