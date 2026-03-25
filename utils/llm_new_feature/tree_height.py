"""
Tree高度计算模块
用于计算LLM生成的表达式树的高度
高度计算规则：采用DEAP的height计算方式
- 叶子节点（特征名）height为0
- 内部节点：height = 1 + max(所有操作数的height)
例如：A+B height为1，max(A+B, C/D) height为2
"""
from typing import Dict, Any, Union


def calculate_tree_height(tree: Union[Dict, str]) -> int:
    """
    计算表达式树的高度（采用DEAP的height计算方式）
    
    高度计算规则：
    - 叶子节点（特征名字符串）：height为0
    - 运算符节点：height = 1 + max(所有操作数的height)
    
    参数:
        tree: 表达式树，可以是：
            - 字符串（特征名）：height为0
            - 字典：{"operator": "...", "operands": [...]}
    
    返回:
        树的高度（整数）
    
    示例:
        >>> calculate_tree_height("SiO2")
        0
        >>> calculate_tree_height({"operator": "Add", "operands": ["SiO2", "Al2O3"]})
        1
        >>> calculate_tree_height({
        ...     "operator": "Max",
        ...     "operands": [
        ...         {"operator": "Add", "operands": ["A", "B"]},
        ...         {"operator": "Div", "operands": ["C", "D"]}
        ...     ]
        ... })
        2
        >>> calculate_tree_height({
        ...     "operator": "Max",
        ...     "operands": [
        ...         {"operator": "Add", "operands": ["A", "B"]},
        ...         {"operator": "Cub", "operands": ["MgO"]}
        ...     ]
        ... })
        2
    """
    if isinstance(tree, str):
        # 叶子节点（特征名）：height为0（DEAP标准）
        return 0
    
    if not isinstance(tree, dict) or "operator" not in tree:
        # 无效的树结构，返回height 0（作为默认值）
        return 0
    
    operands = tree.get("operands", [])
    
    if not operands:
        # 没有操作数的运算符节点：height为0
        return 0
    
    # 计算所有操作数的最大height
    max_operand_height = -1
    for operand in operands:
        operand_height = calculate_tree_height(operand)
        if operand_height > max_operand_height:
            max_operand_height = operand_height
    
    # 运算符节点的height = 1 + 最大操作数height（DEAP标准）
    return 1 + max_operand_height

