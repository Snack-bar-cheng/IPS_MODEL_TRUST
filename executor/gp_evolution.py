"""
GP进化循环模块
负责执行进化算法
"""

import random
import logging
import time
import numpy as np
from deap import tools

# 导入本地工具
import sys
import os
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from evolution_data_saver.generation import save_generation_info
from gp_utils.high_function_expansion import add_high_primitive, expand_high_individual_simple

logger = logging.getLogger(__name__)


def varAnd(population, toolbox, cxpb, mutpb):
    """
    自定义varAnd函数，采用GP-for-Crab-Weight-main项目的策略
    
    该函数实现了概率归一化的交叉变异策略：
    - 对交叉和变异概率进行归一化处理，确保两者互斥
    - 按配对方式处理交叉操作
    - 对相同的配对个体都执行变异
    
    注意：High函数的根节点保护机制已经在toolbox.mate和toolbox.mutate中实现，
    因此该函数会自动继承根节点保护功能。
    
    参数:
        population: 种群列表
        toolbox: DEAP工具箱
        cxpb: 交叉概率
        mutpb: 变异概率
    
    返回:
        经过交叉和变异后的子代种群
    """
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


def run_evolution(population, toolbox, gp_config, evolution_data, 
                  X_train, y_train, feature_names, target_name, 
                  pset=None, llm_features=None):
    """
    执行进化循环
    
    参数:
        population: 初始种群
        toolbox: 工具箱
        gp_config: GP配置
        evolution_data: 进化数据结构
        X_train: 训练特征
        y_train: 训练标签
        feature_names: 特征名称列表
        target_name: 目标变量名称
        pset: 原语集（用于动态分支扩展）
        llm_features: LLM特征列表（用于动态分支扩展）
    
    返回:
        tuple: (population, logbook, training_duration, dynamic_expansion_logs)
    """
    high_func_config = gp_config['high_function']
    enable_high = high_func_config.get('enable', False)
    enable_dynamic_expansion = high_func_config.get('enable_dynamic_expansion', False)
    ngen = gp_config['generations']
    cxpb = gp_config['crossover']['cx_prob']
    mutpb = gp_config['mutation']['mut_prob']
    elitpb = gp_config['elitism_prob']
    
    # 创建统计
    logbook = tools.Logbook()
    
    # 设置统计
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    logbook.header = ["gen", "evals"] + mstats.fields

    # 计算精英群体平均fitness（用于动态扩展判定）
    def compute_elite_avg(pop) -> float:
        k = max(int(elitpb * len(pop)), 1)
        elite = toolbox.selectElitism(pop, k=k)
        if not elite:
            return -np.inf
        vals = [ind.fitness.values[0] if ind.fitness.valid else -np.inf for ind in elite]
        valid = [v for v in vals if v != -np.inf]
        return float(np.mean(valid)) if valid else -np.inf

    # 动态分支扩展配置初始化
    current_high_n = None
    expansion_min_height = None
    expansion_max_height = None
    expansion_llm_ratio = None
    open_dynamic = False
    growth_threshold = 1.0  # 平均增长率阈值（R2*100刻度）
    window_size = None
    cooldown_counter = 0
    elite_fitness_window = []
    
    # ========== 开始计时：核心训练时间（只包括进化过程） ==========
    training_start_time = time.time()
    
    # 评估初始种群（所有个体都评估）
    gen0_start_time = time.time()  # 第0代开始时间
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # 记录第0代
    num_evaluated_gen0 = len(population)  # 第0代评估的个体数
    record = mstats.compile(population)
    logbook.record(gen=0, nevals=num_evaluated_gen0, **record)
    logger.info(logbook.stream)
    logger.info(f"第0代共评估了 {num_evaluated_gen0} 个个体")
    
    # 计算并输出第0代的最大和最小height
    heights_gen0 = [ind.height if hasattr(ind, 'height') else 0 for ind in population]
    if heights_gen0:
        max_height_gen0 = max(heights_gen0)
        min_height_gen0 = min(heights_gen0)
        logger.info(f"第0代height范围: min={max_height_gen0}, max={max_height_gen0}")
    
    # 初始化精英滑动窗口（用于动态扩展判定）
    if enable_high and enable_dynamic_expansion:
        elite_avg_gen0 = compute_elite_avg(population)
        elite_fitness_window.append(elite_avg_gen0)
    
    # 计算第0代的累计时间（从训练开始到第0代统计信息记录完成，不包括保存数据的时间）
    gen0_end_time = time.time()
    gen0_duration = gen0_end_time - training_start_time
    cumulative_time = gen0_duration  # 累计时间（只包括核心训练时间）
    
    # 保存第0代信息（保存数据的时间不计入累计时间）
    if evolution_data is not None:
        enable_high = high_func_config.get('enable', False)
        save_generation_info(0, population, None, record, evolution_data,
                           toolbox=toolbox, train_features=X_train, 
                           train_labels=y_train, feature_names=feature_names,
                           target_variable=target_name, enable_high=enable_high,
                           cumulative_time_seconds=cumulative_time)
    
    # 获取最佳个体用于日志输出
    best_ind_gen0 = max(population, key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
    height_gen0 = best_ind_gen0.height if hasattr(best_ind_gen0, 'height') else 0
    logger.info(f"0 {best_ind_gen0} height={height_gen0}")
    
    # 获取全局最大树高度限制
    max_tree_height = gp_config.get('max_tree_height', 4)
    
    if enable_high and enable_dynamic_expansion:
        current_high_n = high_func_config.get('base_high_n', 3)
        max_high_n = high_func_config.get('max_high_n', 10)
        expansion_interval = high_func_config.get('expansion_interval', 4)
        expansion_min_height = high_func_config.get('expansion_min_height', 1)
        expansion_max_height = high_func_config.get('expansion_max_height', 2)
        expansion_llm_ratio = high_func_config.get('expansion_llm_ratio', 0.5)
        open_dynamic = high_func_config.get('open_dynamic', False)
        growth_threshold = high_func_config.get('growth_threshold', 1.0)
        window_size = max(2, high_func_config.get('window_size', expansion_interval))
        
        # 确保扩展时生成的分支高度不超过全局限制
        # High函数个体的高度 = 1（High根节点） + 最大分支高度
        # 因此分支高度应 <= max_tree_height - 1
        branch_max_height = max_tree_height - 1
        expansion_max_height = min(expansion_max_height, branch_max_height)
    
    # 收集动态分支扩展信息（用于写入txt文件）
    dynamic_expansion_logs = []
    
    # 进化循环
    for gen in range(1, ngen + 1):
        # 记录当代开始时间
        gen_start_time = time.time()
        
        # 更新GWO选择器的代数信息（如果使用GWO）
        if hasattr(toolbox, 'gwo_selector'):
            toolbox.gwo_selector.update_generation(gen, ngen)
        
        # 初始化当代评估计数器
        num_evaluated_this_gen = 0
        
        # 动态分支扩展检查
        if enable_high and enable_dynamic_expansion and pset is not None:
            should_expand = False
            expansion_log = ""
            can_expand = current_high_n < max_high_n
            
            if not open_dynamic:
                # 固定间隔策略
                if gen > expansion_interval and (gen - 1) % expansion_interval == 0 and can_expand:
                    should_expand = True
            else:
                # 自适应策略：窗口满足后，每一代都判断；扩展后进入冷却，再次满 interval 后继续判定
                if can_expand:
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                    else:
                        logger.info(f"[DynCheck] gen={gen} len_window={len(elite_fitness_window)} window_size={window_size} cooldown={cooldown_counter}")
                        # 只要超出expansion_interval，就尝试判定；窗口不足则不扩展也不跳过后续判断
                        if gen > expansion_interval:
                            effective_len = min(len(elite_fitness_window), window_size)
                            if effective_len >= 2:
                                window_slice = elite_fitness_window[-effective_len:]
                                first_val = window_slice[0]
                                last_val = window_slice[-1]
                                prev_val = elite_fitness_window[-2] if len(elite_fitness_window) >= 2 else last_val
                                
                                avg_growth = (last_val - first_val) / (effective_len - 1) if effective_len > 1 else 0.0
                                last_diff = last_val - prev_val
                                
                                if avg_growth > growth_threshold and last_diff > 0:
                                    # 继续探索，不扩展；下一代仍会判断
                                    logger.info(f"[DynCheck] gen={gen} avg_growth={avg_growth:.4f} last_diff={last_diff:.4f} -> 继续探索")
                                else:
                                    should_expand = True
                                    expansion_log = (f"第{gen}代：触发自适应扩展，avg_growth={avg_growth:.4f}, "
                                                     f"last_diff={last_diff:.4f}")
                                    cooldown_counter = expansion_interval
                            else:
                                # 窗口太小无法判断，先不扩展，但继续累积窗口，下一代继续判定
                                logger.info(f"[DynCheck] gen={gen} window不足（len={len(elite_fitness_window)}），跳过扩展继续累积")
            
            if should_expand and can_expand:
                new_high_n = current_high_n + 1
                if not expansion_log:
                    expansion_log = f"第{gen}代开始前：动态分支扩展，从High_{current_high_n}扩展到High_{new_high_n}"
                logger.info(expansion_log)
                
                # 添加新的High_N原语
                if add_high_primitive(pset, new_high_n):
                    # 扩展种群中使用旧High函数的个体
                    expanded_count = 0
                    # 统计新增分支的来源（LLM vs Random）
                    total_new_branches = 0
                    llm_branches_count = 0
                    random_branches_count = 0
                    
                    def gen_llm_expr():
                        if llm_features and feature_names:
                            feature = random.choice(llm_features)
                            try:
                                # 从本地utils目录导入
                                utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
                                if utils_path not in sys.path:
                                    sys.path.insert(0, utils_path)
                                from llm_new_feature.llm_to_gp_converter import llm_tree_to_gp_expr
                                # 提取tree字段（LLM特征对象包含tree、description、notation等字段）
                                tree_dict = feature.get("tree", {}) if isinstance(feature, dict) else feature
                                return llm_tree_to_gp_expr(tree_dict, pset, feature_names)
                            except:
                                # 从本地模块导入树生成函数
                                from gp_utils.tree_generation import gp_restrict
                                Float1 = float
                                return gp_restrict.genHalfAndHalfMD(pset, expansion_min_height, expansion_max_height, Float1)
                        return None
                    
                    for i, ind in enumerate(population):
                        ind_expr = list(ind)
                        has_old_high = any(
                            hasattr(node, 'name') and node.name == f'High_{current_high_n}'
                            for node in ind_expr
                        )
                        
                        if has_old_high:
                            try:
                                expanded_ind, branch_stats = expand_high_individual_simple(
                                    ind, pset, toolbox, current_high_n, new_high_n,
                                    llm_features=llm_features, 
                                    feature_names=feature_names,
                                    gen_llm_expr=gen_llm_expr,
                                    expansion_min_height=expansion_min_height,
                                    expansion_max_height=expansion_max_height,
                                    expansion_llm_ratio=expansion_llm_ratio
                                )
                                
                                # 收集分支统计信息
                                if branch_stats is not None:
                                    total_new_branches += (branch_stats.get('llm_count', 0) + branch_stats.get('random_count', 0))
                                    llm_branches_count += branch_stats.get('llm_count', 0)
                                    random_branches_count += branch_stats.get('random_count', 0)
                                
                                # 检查扩展是否成功
                                expanded_expr = list(expanded_ind)
                                has_new_high = any(
                                    hasattr(node, 'name') and node.name == f'High_{new_high_n}'
                                    for node in expanded_expr
                                )
                                
                                if has_new_high:
                                    population[i] = expanded_ind
                                    expanded_count += 1
                            except Exception as e:
                                logger.warning(f"扩展个体{i}时出错: {e}")
                    
                    expanded_log = f"成功扩展 {expanded_count}/{len(population)} 个个体"
                    logger.info(expanded_log)
                    expansion_log += "\n" + expanded_log
                    
                    if total_new_branches > 0:
                        branch_stats_log = f"新增分支统计: LLM特征 {llm_branches_count} 个, 随机特征 {random_branches_count} 个 (总计 {total_new_branches} 个)"
                        logger.info(branch_stats_log)
                        expansion_log += "\n" + branch_stats_log
                    
                    # 保存扩展日志
                    dynamic_expansion_logs.append(expansion_log)
                    
                    # 重新评估扩展后的个体（所有扩展的个体都重新评估）
                    if expanded_count > 0:
                        # 找到所有被扩展的个体并重新评估
                        expanded_individuals = []
                        for i, ind in enumerate(population):
                            ind_expr = list(ind)
                            has_new_high = any(
                                hasattr(node, 'name') and node.name == f'High_{new_high_n}'
                                for node in ind_expr
                            )
                            if has_new_high:
                                expanded_individuals.append(ind)
                        
                        if expanded_individuals:
                            num_evaluated_expansion = len(expanded_individuals)
                            num_evaluated_this_gen += num_evaluated_expansion
                            fitnesses = toolbox.map(toolbox.evaluate, expanded_individuals)
                            for ind, fit in zip(expanded_individuals, fitnesses):
                                ind.fitness.values = fit
                    
                    current_high_n = new_high_n
        
        # 精英保留：从上一代population中选择精英（用于保留到当前代）
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        elite_from_prev_gen = toolbox.selectElitism(population_for_eli, k=elitismNum)
        
        # 选择剩余个体（用于交叉变异）
        offspring = toolbox.select(population, len(population) - elitismNum)
        
        # 交叉和变异
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
        # 评估所有offspring个体（不进行缓存，所有个体都评估）
        num_evaluated_offspring = len(offspring)
        num_evaluated_this_gen += num_evaluated_offspring
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        # 重新评估保留的精英个体（因为环境可能变化，如动态扩展）
        if len(elite_from_prev_gen) > 0:
            num_evaluated_elite = len(elite_from_prev_gen)
            num_evaluated_this_gen += num_evaluated_elite
            elite_fitnesses = toolbox.map(toolbox.evaluate, elite_from_prev_gen)
            for ind, fit in zip(elite_from_prev_gen, elite_fitnesses):
                ind.fitness.values = fit
        
        # 合并：保留的精英 + 新产生的offspring = 当前代的完整population
        offspring.extend(elite_from_prev_gen)
        
        # 更新种群为当前代的所有个体（包括保留的精英和新产生的offspring）
        population[:] = offspring
        
        # 注意：下一代的精英选择会在下一代的循环开始时（第281行）从当前代的population中选择
        # 这样确保每代的精英都是从当前代的所有个体（包括新产生的offspring）中选择的真正最好的个体
        
        # 记录当前代
        record = mstats.compile(population)
        logbook.record(gen=gen, nevals=num_evaluated_this_gen, **record)
        logger.info(logbook.stream)
        logger.info(f"第{gen}代共评估了 {num_evaluated_this_gen} 个个体")
        
        # 计算并输出当前代的最大和最小height
        heights = [ind.height if hasattr(ind, 'height') else 0 for ind in population]
        if heights:
            max_height = max(heights)
            min_height = min(heights)
            logger.info(f"第{gen}代height范围: min={min_height}, max={max_height}")
        
        # 更新精英滑动窗口（用于下一代的动态扩展判定）
        if enable_high and enable_dynamic_expansion:
            elite_avg_curr = compute_elite_avg(population)
            elite_fitness_window.append(elite_avg_curr)
            if len(elite_fitness_window) > window_size:
                elite_fitness_window.pop(0)
            
            # 输出窗口内的平均增长率与最近一代边际增益
            if len(elite_fitness_window) >= 2:
                effective_len = min(len(elite_fitness_window), window_size)
                window_slice = elite_fitness_window[-effective_len:]
                first_val = window_slice[0]
                last_val = window_slice[-1]
                prev_val = elite_fitness_window[-2]
                avg_growth = (last_val - first_val) / (effective_len - 1) if effective_len > 1 else 0.0
                last_diff = last_val - prev_val
                logger.info(f"[DynWindow] gen={gen} window={window_slice} avg_growth={avg_growth:.4f} last_diff={last_diff:.4f}")
            else:
                logger.info(f"[DynWindow] gen={gen} window={elite_fitness_window}")
        
        # 记录当代结束时间（在保存代信息之前，这样累计时间只包括核心训练时间）
        gen_end_time = time.time()
        gen_duration = gen_end_time - gen_start_time
        # 累计时间 = 上一代的累计时间 + 当代的训练时间
        cumulative_time += gen_duration
        
        # 保存当前代信息（累计时间不包括保存数据的时间，以匹配 training_duration 的定义）
        if evolution_data is not None:
            save_generation_info(gen, population, None, record, evolution_data,
                               toolbox=toolbox, train_features=X_train,
                               train_labels=y_train, feature_names=feature_names,
                               target_variable=target_name, enable_high=enable_high,
                               cumulative_time_seconds=cumulative_time)
        
        # 获取最佳个体用于日志输出
        best_ind = max(population, key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
        height = best_ind.height if hasattr(best_ind, 'height') else 0
        logger.info(f"{gen} {best_ind} height={height}")
    
    # ========== 结束计时：核心训练时间 ==========
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    # 验证最后一代的累计时间是否与 training_duration 一致
    # 理论上应该非常接近，因为两者都只包括核心训练时间
    if abs(cumulative_time - training_duration) > 0.1:  # 允许0.1秒的误差
        logger.warning(f"累计时间 ({cumulative_time:.4f}s) 与训练总时间 ({training_duration:.4f}s) 不一致，差异: {abs(cumulative_time - training_duration):.4f}s")
    
    logger.info(f"核心训练时间: {training_duration:.4f} 秒 ({training_duration/60:.2f} 分钟)")
    
    return population, logbook, training_duration, dynamic_expansion_logs

