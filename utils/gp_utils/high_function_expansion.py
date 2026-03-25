"""
High函数动态扩展工具
"""

import random
import logging
import numpy as np
from deap import gp, creator

# 从本地模块导入树生成函数
from .tree_generation import gp_restrict

# 从本地模块导入特征组合函数
from strategies.function_set.feature_combination_strategies import (
    root_con, root_con1, root_con2, root_con3, root_con4, root_con5, root_con6, 
    root_con7, root_con8, root_con9, root_con10
)

# 定义类型
Float1 = float
Vector1 = np.ndarray

logger = logging.getLogger(__name__)


def add_high_primitive(pset, n):
    """
    动态添加High_N原语到pset
    
    参数:
        pset: 原语集
        n: High函数的参数数量（分支数）
    
    返回:
        bool: 是否成功添加
    """
    primitive_name = f'High_{n}'
    
    # 检查是否已经存在
    if Vector1 in pset.primitives:
        for p in pset.primitives[Vector1]:
            if p.name == primitive_name:
                logger.debug(f"原语 {primitive_name} 已存在，跳过添加")
                return False
    
    # 根据n的值选择对应的函数
    if n == 1:
        func = root_con1
        arg_types = [Float1]
    elif n == 2:
        func = root_con2
        arg_types = [Float1, Float1]
    elif n == 3:
        func = root_con3
        arg_types = [Float1, Float1, Float1]
    elif n == 4:
        func = root_con4
        arg_types = [Float1, Float1, Float1, Float1]
    elif n == 5:
        func = root_con5
        arg_types = [Float1, Float1, Float1, Float1, Float1]
    elif n == 6:
        func = root_con6
        arg_types = [Float1, Float1, Float1, Float1, Float1, Float1]
    elif n == 7:
        func = root_con7
        arg_types = [Float1, Float1, Float1, Float1, Float1, Float1, Float1]
    elif n == 8:
        func = root_con8
        arg_types = [Float1, Float1, Float1, Float1, Float1, Float1, Float1, Float1]
    elif n == 9:
        func = root_con9
        arg_types = [Float1, Float1, Float1, Float1, Float1, Float1, Float1, Float1, Float1]
    elif n == 10:
        func = root_con10
        arg_types = [Float1, Float1, Float1, Float1, Float1, Float1, Float1, Float1, Float1, Float1]
    else:
        logger.warning(f"不支持的分支数: {n}，支持范围是1-10")
        return False
    
    # 添加原语
    pset.addPrimitive(func, arg_types, Vector1, name=primitive_name)
    logger.info(f"成功添加原语: {primitive_name}")
    return True


def expand_high_individual_simple(individual, pset, toolbox, old_n, new_n, 
                                  llm_features=None, feature_names=None, gen_llm_expr=None,
                                  expansion_min_height=1, expansion_max_height=2,
                                  expansion_llm_ratio=0.5):
    """
    简化版本的扩展函数：使用GP树遍历来提取和替换High节点
    
    参数:
        individual: 要扩展的个体
        pset: 原语集
        toolbox: 工具箱
        old_n: 旧的High函数分支数
        new_n: 新的High函数分支数
        llm_features: LLM特征列表（可选）
        feature_names: 特征名称列表（可选）
        gen_llm_expr: LLM表达式生成函数（可选）
        expansion_min_height: 动态扩展时新生成分支的最小高度（默认1）
        expansion_max_height: 动态扩展时新生成分支的最大高度（默认2）
        expansion_llm_ratio: 动态扩展时新生成分支中LLM特征的比例（0-1之间，默认0.5）
                            0表示不使用LLM特征，0.1表示10%使用LLM特征
    
    返回:
        tuple: (new_individual, branch_stats)
            - new_individual: 扩展后的新个体
            - branch_stats: dict，包含 {'llm_count': int, 'random_count': int}
    """
    # 获取新的High原语
    new_high_prim = None
    if Vector1 in pset.primitives:
        for p in pset.primitives[Vector1]:
            if p.name == f'High_{new_n}':
                new_high_prim = p
                break
    
    if new_high_prim is None:
        logger.warning(f"未找到原语 High_{new_n}，无法扩展个体")
        return toolbox.clone(individual), None
    
    try:
        old_expr = list(individual)
        new_expr = []
        i = 0
        branch_stats = {'llm_count': 0, 'random_count': 0}
        
        # 提取完整子树的辅助函数
        def extract_subtree(expr_list, start_pos):
            """提取从start_pos开始的完整子树"""
            if start_pos is None or not isinstance(start_pos, int) or start_pos < 0:
                return None, 0
            if expr_list is None or start_pos >= len(expr_list):
                return None, start_pos if start_pos is not None else 0
            
            subtree = []
            pos = int(start_pos)
            stack = [1]
            
            while stack and pos < len(expr_list):
                node = expr_list[pos]
                subtree.append(node)
                
                arity = 0
                if hasattr(node, 'arity'):
                    try:
                        arity_val = node.arity
                        if arity_val is not None:
                            arity = int(arity_val) if isinstance(arity_val, (int, float)) else 0
                    except:
                        arity = 0
                
                if stack:
                    stack[-1] -= 1
                    if arity > 0:
                        stack.append(int(arity))
                
                while stack and stack[-1] == 0:
                    stack.pop()
                
                pos += 1
                if not stack:
                    break
            
            return subtree, pos
        
        # 遍历表达式，查找并替换High_N节点
        found_high = False
        while i < len(old_expr):
            node = old_expr[i]
            
            if hasattr(node, 'name') and node.name == f'High_{old_n}':
                found_high = True
                branches = []
                branch_pos = i + 1
                
                # 提取old_n个分支
                for branch_idx in range(old_n):
                    if branch_pos >= len(old_expr):
                        break
                    branch, next_pos = extract_subtree(old_expr, branch_pos)
                    if branch and len(branch) > 0 and next_pos > branch_pos:
                        branches.append(branch)
                        branch_pos = next_pos
                    else:
                        break
                
                if len(branches) == old_n:
                    new_expr.append(new_high_prim)
                    
                    # 添加原有分支
                    for branch in branches:
                        new_expr.extend(branch)
                    
                    # 生成新的分支
                    for branch_idx in range(new_n - old_n):
                        is_llm_branch = False
                        # 使用配置的LLM比例来决定是否使用LLM特征
                        if llm_features and feature_names and gen_llm_expr and expansion_llm_ratio > 0 and random.random() < expansion_llm_ratio:
                            try:
                                new_branch = gen_llm_expr()
                                if new_branch:
                                    is_llm_branch = True
                                    branch_stats['llm_count'] += 1
                                else:
                                    new_branch = gp_restrict.genHalfAndHalfMD(pset, expansion_min_height, expansion_max_height, Float1)
                                    branch_stats['random_count'] += 1
                            except Exception as e:
                                logger.debug(f"LLM生成异常: {e}，回退到随机生成")
                                new_branch = gp_restrict.genHalfAndHalfMD(pset, expansion_min_height, expansion_max_height, Float1)
                                branch_stats['random_count'] += 1
                        else:
                            new_branch = gp_restrict.genHalfAndHalfMD(pset, expansion_min_height, expansion_max_height, Float1)
                            branch_stats['random_count'] += 1
                        new_expr.extend(new_branch)
                    
                    if branch_pos > i:
                        i = branch_pos
                    else:
                        i += 1
                else:
                    new_expr.append(node)
                    i += 1
            else:
                new_expr.append(node)
                i += 1
        
        if not found_high:
            logger.warning(f"个体中未找到High_{old_n}节点，无法扩展")
            return toolbox.clone(individual), None
        
        # 创建新个体
        try:
            new_ind = creator.Individual(new_expr)
            new_ind.is_llm_generated = getattr(individual, 'is_llm_generated', False)
            del new_ind.fitness.values
        except Exception as e:
            logger.warning(f"创建新个体时出错: {e}")
            new_ind = toolbox.clone(individual)
            while len(new_ind) > 0:
                new_ind.pop()
            for node in new_expr:
                new_ind.append(node)
            new_ind.is_llm_generated = getattr(individual, 'is_llm_generated', False)
            del new_ind.fitness.values
        
        return new_ind, branch_stats
        
    except Exception as e:
        logger.warning(f"扩展个体时出错: {e}")
        return toolbox.clone(individual), None

