"""
GP系统设置模块
负责设置原语集、工具箱等
"""

import random
import logging
import warnings
import numpy as np
from deap import base, creator, tools, gp
import operator

# 忽略numpy和sklearn的运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')
np.seterr(all='ignore')  # 忽略所有numpy的浮点错误

# 从本地模块导入所有需要的函数
from strategies.function_set.feature_combination_strategies import (
    root_con, root_con1, root_con2, root_con3, root_con4, root_con5, root_con6,
    root_con7, root_con8, root_con9, root_con10
)
from strategies.function_set.arithmetic_operators import (
    addition, subtraction, multiplication, division,
    maximum, minimum, mean,
    log_transform, log10_transform,
    square, cube, square_root, cube_root
)

# 导入本地工具 - 先设置路径，再导入
import sys
import os
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from llm_new_feature.llm_to_gp_converter import load_llm_features, llm_tree_to_gp_expr
from gp_utils.tree_generation import gp_restrict
from gp_utils.fitness import fitness_high_gp, eval_traditional_gp, which_fitness_to_use
from gp_utils.high_function_expansion import add_high_primitive

# 定义类型系统
Float1 = float
Vector1 = np.ndarray

logger = logging.getLogger(__name__)


def setup_primitive_set(X_train, feature_names, func_config, high_func_config):
    """
    设置原语集
    
    参数:
        X_train: 训练特征
        feature_names: 特征名称列表
        func_config: 函数集配置（包含operators列表）
        high_func_config: High函数配置
    
    返回:
        pset: 原语集
    """
    enable_high = high_func_config.get('enable', False)
    operators = func_config.get('operators', [])
    
    # 根据是否启用High函数选择返回类型
    if enable_high:
        return_type = Vector1
    else:
        return_type = Float1
    
    # 创建原语集
    pset = gp.PrimitiveSetTyped('MAIN', 
                                [Float1] * X_train.shape[1], 
                                return_type)
    
    # 重命名变量
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f'ARG{i}': name})
    
    # 注册High函数（仅在启用时）
    if enable_high:
        enable_dynamic_expansion = high_func_config.get('enable_dynamic_expansion', False)
        
        if enable_dynamic_expansion:
            # 动态扩展模式：使用 base_high_n 作为初始注册
            base_high_n = high_func_config.get('base_high_n', 3)
            high_n_list = [base_high_n]
        else:
            # 静态模式：使用 high_n 配置
            high_n_config = high_func_config.get('high_n', 3)
            if isinstance(high_n_config, int):
                high_n_list = [high_n_config]
            elif isinstance(high_n_config, list):
                high_n_list = high_n_config
            else:
                high_n_list = [3]  # 默认值
        
        # 为每个 high_n 注册对应的 High 函数（通用逻辑）
        registered_high_functions = []
        for n in high_n_list:
            # 为每个n值选择对应的函数
            if n == 1:
                high_func = root_con1
            elif n == 2:
                high_func = root_con2
            elif n == 3:
                high_func = root_con3
            elif n == 4:
                high_func = root_con4
            elif n == 5:
                high_func = root_con5
            elif n == 6:
                high_func = root_con6
            elif n == 7:
                high_func = root_con7
            elif n == 8:
                high_func = root_con8
            elif n == 9:
                high_func = root_con9
            elif n == 10:
                high_func = root_con10
            else:
                # 对于其他n值，使用lambda创建函数
                def make_high_func(num):
                    def high_n_func(*args):
                        if len(args) != num:
                            raise ValueError(f"High_{num}期望{num}个参数，但得到{len(args)}个")
                        return root_con(*args)
                    return high_n_func
                high_func = make_high_func(n)
            
            pset.addPrimitive(high_func, 
                            [Float1]*n, Vector1, 
                            name=f'High_{n}')
            registered_high_functions.append(f'High_{n}')
        
        # 验证High函数是否已正确注册到pset中
        if Vector1 in pset.primitives:
            actual_high_functions = [p.name for p in pset.primitives[Vector1] if p.name.startswith('High_')]
            logger.info(f"已注册的High函数列表: {sorted(actual_high_functions)}")
            if not actual_high_functions:
                logger.warning("⚠️ 警告：pset中没有找到任何High函数！")
        else:
            logger.warning("⚠️ 警告：pset中没有Vector1类型的原语！")
    
    # 注册运算符（从operators列表中读取）
    if 'Add' in operators:
        pset.addPrimitive(addition, [Float1, Float1], Float1, name="Add")
    if 'Sub' in operators:
        pset.addPrimitive(subtraction, [Float1, Float1], Float1, name="Sub")
    if 'Mul' in operators:
        pset.addPrimitive(multiplication, [Float1, Float1], Float1, name="Mul")
    if 'Div' in operators:
        pset.addPrimitive(division, [Float1, Float1], Float1, name='Div')
    if 'Max' in operators:
        pset.addPrimitive(maximum, [Float1, Float1], Float1, name='Max')
    if 'Min' in operators:
        pset.addPrimitive(minimum, [Float1, Float1], Float1, name='Min')
    if 'Mean' in operators:
        pset.addPrimitive(mean, [Float1, Float1], Float1, name='Mean')
    if 'Ln' in operators:
        pset.addPrimitive(log_transform, [Float1], Float1, name='Ln')
    if 'Log' in operators:
        pset.addPrimitive(log10_transform, [Float1], Float1, name='Log')
    if 'Squ' in operators:
        pset.addPrimitive(square, [Float1], Float1, name='Squ')
    if 'Cub' in operators:
        pset.addPrimitive(cube, [Float1], Float1, name='Cub')
    if 'Sqrt' in operators:
        pset.addPrimitive(square_root, [Float1], Float1, name='Sqrt')
    if 'Cbrt' in operators:
        pset.addPrimitive(cube_root, [Float1], Float1, name='Cbrt')
    
    # 注册临时常数
    ephemeral_config = func_config.get('enable_ephemeral_constant', [0, 3000])
    if ephemeral_config is not False:
        if isinstance(ephemeral_config, list) and len(ephemeral_config) == 2:
            min_val, max_val = ephemeral_config
            pset.addEphemeralConstant("RandFloat", 
                                    lambda: random.uniform(min_val, max_val), Float1)
        elif ephemeral_config is True:
            # 兼容旧配置：True 表示使用默认范围
            pset.addEphemeralConstant("RandFloat", 
                                    lambda: random.uniform(0, 3000), Float1)
    
    return pset


def setup_toolbox(pset, X_train, y_train, gp_config, feature_names, llm_features=None):
    """
    设置工具箱
    
    参数:
        pset: 原语集
        X_train: 训练特征
        y_train: 训练标签
        gp_config: GP配置
        feature_names: 特征名称列表
        llm_features: LLM特征列表（可选）
    
    返回:
        toolbox: 工具箱对象
    """
    init_config = gp_config['initialization']
    high_func_config = gp_config['high_function']
    enable_high = high_func_config.get('enable', False)
    
    # 创建工具箱
    toolbox = base.Toolbox()
    
    # 获取全局最大树高度限制（参考GP-for-Crab-Weight-main的maxDepth策略）
    max_tree_height = gp_config.get('max_tree_height', 4)  # 默认值4，与GP-for-Crab-Weight-main一致
    
    # 注册初始化策略（兼容旧配置中的depth）
    random_min = init_config['random'].get('initial_min_height', init_config['random'].get('initial_min_depth', 1))
    random_max_original = init_config['random'].get('initial_max_height', init_config['random'].get('initial_max_depth', 2))
    
    # 记录配置值用于调试
    logger.info(f"初始化配置 - max_tree_height: {max_tree_height}, initial_max_height: {random_max_original}, enable_high: {enable_high}")
    
    # 确保初始化高度不超过全局限制
    if enable_high:
        # High函数模式：分支高度应 <= max_tree_height - 1
        branch_max_height = max_tree_height - 1
        random_max = min(random_max_original, branch_max_height)
        return_type = Vector1
        logger.info(f"High函数模式 - branch_max_height: {branch_max_height}, 最终random_max: {random_max}")
    else:
        # 传统GP模式：直接使用全局限制
        random_max = min(random_max_original, max_tree_height)
        return_type = Float1
        logger.info(f"传统GP模式 - 最终random_max: {random_max} (min({random_max_original}, {max_tree_height}))")
    
    def genRandomExpr():
        return gp_restrict.genHalfAndHalfMD(pset, random_min, random_max, return_type)
    toolbox.register("expr_random", genRandomExpr)
    
    if init_config['llm']['enabled'] and llm_features:
        def genLLMExpr():
            if not llm_features:
                return genRandomExpr()
            
            # 如果返回类型是Vector1，需要生成High_N表达式
            if return_type == Vector1:
                # 查找所有可用的High_N原语
                available_primitives = []
                if Vector1 in pset.primitives:
                    for p in pset.primitives[Vector1]:
                        if p.name.startswith('High_'):
                            try:
                                n = int(p.name.split('_')[1])
                                available_primitives.append((n, p))
                            except (ValueError, IndexError):
                                continue
                
                # 随机选择一个可用的High_N原语
                if available_primitives:
                    feature_count, high_prim = random.choice(available_primitives)
                    logger.debug(f"随机选择High_N原语: {high_prim.name}，将生成 {feature_count} 个LLM特征")
                else:
                    logger.warning("未找到任何High_N原语，回退到随机生成")
                    return genRandomExpr()
                
                # 生成N个LLM特征表达式
                llm_exprs = []
                max_attempts_per_feature = min(len(llm_features), 50)
                attempted_indices = set()
                
                for feature_idx in range(feature_count):
                    feature_expr = None
                    feature_attempted = set()
                    
                    for attempt in range(max_attempts_per_feature):
                        # 随机选择一个未尝试过的LLM特征
                        available_indices = [i for i in range(len(llm_features)) 
                                            if i not in attempted_indices and i not in feature_attempted]
                        if not available_indices:
                            available_indices = [i for i in range(len(llm_features)) 
                                                if i not in attempted_indices]
                            if not available_indices:
                                available_indices = list(range(len(llm_features)))
                        
                        selected_idx = random.choice(available_indices)
                        feature_attempted.add(selected_idx)
                        selected_feature = llm_features[selected_idx]
                        tree_dict = selected_feature.get("tree", {}) if isinstance(selected_feature, dict) else selected_feature
                        
                        if not tree_dict:
                            continue
                        
                        try:
                            # 转换为GP表达式（返回Float1类型）
                            expr = llm_tree_to_gp_expr(tree_dict, pset, feature_names)
                            if expr and len(expr) > 0:
                                has_primitive = any(hasattr(item, 'arity') for item in expr)
                                has_terminal = any(isinstance(item, gp.Terminal) for item in expr)
                                if has_primitive or has_terminal:
                                    feature_expr = expr
                                    attempted_indices.add(selected_idx)
                                    break
                        except:
                            continue
                    
                    if feature_expr is None:
                        # 如果无法生成LLM特征，使用随机生成作为后备
                        # 对于High函数模式，分支高度应 <= max_tree_height - 1
                        if enable_high:
                            branch_max_height = max_tree_height - 1
                            feature_expr = gp_restrict.genHalfAndHalfMD(pset, random_min, min(random_max, branch_max_height), Float1)
                        else:
                            feature_expr = gp_restrict.genHalfAndHalfMD(pset, random_min, random_max, Float1)
                    
                    llm_exprs.append(feature_expr)
                
                # 构建High_N表达式：[High_N, expr1, expr2, ..., exprN]
                high_n_expr = [high_prim]
                for expr in llm_exprs:
                    high_n_expr.extend(expr)
                
                return high_n_expr
            else:
                # 如果返回类型是Float1，返回单个LLM特征
                feature = random.choice(llm_features)
                try:
                    tree_dict = feature.get("tree", {}) if isinstance(feature, dict) else feature
                    expr = llm_tree_to_gp_expr(tree_dict, pset, feature_names)
                    return expr
                except:
                    return genRandomExpr()
        toolbox.register("expr_llm", genLLMExpr)
    
    # 注册个体创建函数
    def create_random_individual():
        expr = toolbox.expr_random()
        ind = creator.Individual(expr)
        ind.is_llm_generated = False
        return ind
    
    def create_llm_individual():
        if init_config['llm']['enabled'] and llm_features:
            expr = toolbox.expr_llm()
            ind = creator.Individual(expr)
            ind.is_llm_generated = True
            return ind
        return create_random_individual()
    
    toolbox.register("individual_random", create_random_individual)
    if init_config['llm']['enabled'] and llm_features:
        toolbox.register("individual_llm", create_llm_individual)
    
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("map", map)
    
    # 注册适应度评估
    # 使用which_fitness_to_use自动根据个体类型选择适应的适应度函数
    # 使用固定随机种子42，确保每次执行时交叉验证划分一致
    cv_random_state = gp_config.get('cv_random_state', 42)
    def evalSymbReg(individual):
        return which_fitness_to_use(
            individual, toolbox, X_train, y_train,
            cv_folds=gp_config['cv_folds'],
            scale_factor=gp_config['scale_factor'],
            random_state=cv_random_state
        )
    
    toolbox.register("evaluate", evalSymbReg)
    
    # 注册选择算子
    selection_config = gp_config['selection']
    enable_gwo = selection_config.get("enable_gwo", False)
    
    if enable_gwo:
        # 获取GWO类型（传统或改进）
        gwo_type = selection_config.get("gwo_type", "enhanced")  # 默认为改进GWO
        
        # 获取最大代数用于GWO
        max_generations = gp_config.get('generations', 50)
        
        if gwo_type == "traditional":
            # 使用传统GWO算法
            from strategies.gwo import TraditionalGWO
            
            gwo_selector = TraditionalGWO()
            gwo_selector.update_generation(0, max_generations)
            
            # 注册传统GWO选择算子
            def gwo_select(population, k):
                return gwo_selector.select(population, k)
            
            toolbox.register("select", gwo_select)
            toolbox.gwo_selector = gwo_selector  # 保存引用以便在进化循环中更新代数
            logger.info("已启用传统GWO选择算法")
        else:
            # 使用改进的GWO算法
            from strategies.gwo import EnhancedGWO, EnhancedGWOConfig
            
            gwo_config_dict = selection_config.get("gwo_config", {})
            gwo_config = EnhancedGWOConfig(gwo_config_dict)
            gwo_selector = EnhancedGWO(gwo_config)
            gwo_selector.update_generation(0, max_generations)
            
            # 注册改进GWO选择算子
            def gwo_select(population, k):
                return gwo_selector.select(population, k)
            
            toolbox.register("select", gwo_select)
            toolbox.gwo_selector = gwo_selector  # 保存引用以便在进化循环中更新代数
            logger.info("已启用改进的GWO选择算法")
    else:
        # 使用传统的锦标赛选择
        tournament_size = selection_config.get('tournament_size', 7)
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        logger.info(f"使用锦标赛选择，tournament_size={tournament_size}")
    
    toolbox.register("selectElitism", tools.selBest)
    
    # 获取全局最大树高度限制（参考GP-for-Crab-Weight-main的maxDepth策略）
    max_tree_height = gp_config.get('max_tree_height', 4)  # 默认值4，与GP-for-Crab-Weight-main一致
    
    # 变异子树生成参数
    mut_min = gp_config['mutation']['mutate_gen_full_min']
    mut_max = gp_config['mutation']['mutate_gen_full_max']
    
    # 对于High函数模式，需要自定义高度计算函数
    # High函数个体的高度 = 1（High根节点） + 最大分支高度
    # 分支高度应 <= max_tree_height - 1
    def calculate_high_individual_height(individual):
        """
        计算High函数个体的高度
        高度 = 1（High根节点） + 最大分支高度
        """
        if len(individual) == 0:
            return 0
        
        root_node = individual[0]
        is_high_root = hasattr(root_node, 'name') and root_node.name and root_node.name.startswith('High_')
        
        if not is_high_root:
            # 如果不是High函数，使用标准高度计算
            return individual.height
        
        # 提取所有分支（从索引1开始）
        branches = []
        i = 1
        while i < len(individual):
            # 提取一个完整的分支子树
            branch_start = i
            stack = [1]  # 从当前节点开始
            
            while stack and i < len(individual):
                node = individual[i]
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
                
                i += 1
                
                while stack and stack[-1] == 0:
                    stack.pop()
                
                if not stack:
                    break
            
            # 创建分支个体并计算高度
            if branch_start < i:
                branch_expr = list(individual[branch_start:i])
                if branch_expr:
                    try:
                        branch_ind = creator.Individual(branch_expr)
                        branch_height = branch_ind.height
                        branches.append(branch_height)
                    except:
                        # 如果无法创建个体，跳过这个分支
                        pass
        
        # 返回 1（High根节点） + 最大分支高度
        if branches:
            max_branch_height = max(branches)
            return 1 + max_branch_height
        else:
            return 1  # 只有根节点，没有分支
    
    if enable_high:
        # High函数模式：保护High根节点
        # 分支高度限制：max_tree_height - 1（因为High根节点占1层）
        branch_max_height = max_tree_height - 1
        
        def mate_preserve_high_root(ind1, ind2):
            """保护High根节点的交叉"""
            if len(ind1) == 0 or len(ind2) == 0:
                return (ind1, ind2)
            root1, root2 = ind1[0], ind2[0]
            is_high1 = hasattr(root1, 'name') and root1.name and root1.name.startswith('High_')
            is_high2 = hasattr(root2, 'name') and root2.name and root2.name.startswith('High_')
            
            if is_high1 and is_high2 and len(ind1) > 1 and len(ind2) > 1:
                branch1 = creator.Individual(ind1[1:])
                branch2 = creator.Individual(ind2[1:])
                branch1, branch2 = gp.cxOnePoint(branch1, branch2)
                new_ind1 = creator.Individual([root1] + list(branch1))
                new_ind2 = creator.Individual([root2] + list(branch2))
                new_ind1.is_llm_generated = getattr(ind1, 'is_llm_generated', False)
                new_ind2.is_llm_generated = getattr(ind2, 'is_llm_generated', False)
                del new_ind1.fitness.values, new_ind2.fitness.values
                return (new_ind1, new_ind2)
            return gp.cxOnePoint(ind1, ind2)
        
        def mutate_preserve_high_root(individual, expr_mut, pset):
            """保护High根节点的变异"""
            if len(individual) == 0:
                return (individual,)
            root_node = individual[0]
            is_high_root = hasattr(root_node, 'name') and root_node.name and root_node.name.startswith('High_')
            
            if is_high_root and len(individual) > 1:
                branch_ind = creator.Individual(individual[1:])
                branch_ind, = gp.mutUniform(branch_ind, expr_mut, pset)
                new_ind = creator.Individual([root_node] + list(branch_ind))
                new_ind.is_llm_generated = getattr(individual, 'is_llm_generated', False)
                del new_ind.fitness.values
                return (new_ind,)
            return gp.mutUniform(individual, expr_mut, pset)
        
        toolbox.register("mate", mate_preserve_high_root)
        # 变异子树生成：限制分支高度不超过 branch_max_height
        toolbox.register("expr_mut", gp_restrict.genFull, min_=mut_min, max_=min(mut_max, branch_max_height))
        toolbox.register("mutate", mutate_preserve_high_root, 
                        expr_mut=toolbox.expr_mut, pset=pset)
        
        # 使用自定义高度计算函数进行限制
        toolbox.decorate("mate", gp.staticLimit(key=calculate_high_individual_height, max_value=max_tree_height))
        toolbox.decorate("mutate", gp.staticLimit(key=calculate_high_individual_height, max_value=max_tree_height))
    else:
        # 传统GP模式：标准交叉和变异
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp_restrict.genFull, min_=mut_min, max_=mut_max)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        
        # 使用全局高度限制（参考GP-for-Crab-Weight-main的maxDepth策略）
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))
    
    return toolbox

