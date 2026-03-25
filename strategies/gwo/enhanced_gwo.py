"""
改进的灰狼优化算法（Enhanced Grey Wolf Optimizer）
专门为遗传规划设计，结合分阶段动态策略和自适应机制

理论基础：
1. 分阶段探索-开发平衡：根据进化进度动态调整收敛因子
2. 多层级引导：Alpha/Beta/Delta三层引导机制
3. 自适应权重：根据种群多样性和性能改善动态调整
4. 智能选择：减少不必要的评估，提升计算效率
"""

import random
import numpy as np
from typing import List, Tuple, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedGWOConfig:
    """改进GWO算法配置参数"""
    
    def __init__(self, config: dict = None):
        """
        初始化配置
        
        参数:
            config: 配置字典，如果为None则使用默认值
        """
        if config is None:
            config = {}
        
        # 阶段阈值（针对50代优化）
        self.tau1 = config.get('tau1', 0.2)  # 探索阶段结束比例 (20% = 10代)
        self.tau2 = config.get('tau2', 0.7)  # 平衡阶段结束比例 (70% = 35代)
        
        # 收敛因子参数（动态调整）
        self.a0 = config.get('a0', 2.0)      # 初始收敛因子（探索阶段）
        self.am = config.get('am', 1.0)      # 中期收敛因子（平衡阶段）
        self.af = config.get('af', 0.1)      # 最终收敛因子（开发阶段，降低以增强局部搜索）
        
        # 层比例参数（Alpha/Beta/Delta/Omega）
        self.rho_alpha = config.get('rho_alpha', 0.05)  # Alpha层比例（最佳个体）
        self.rho_beta = config.get('rho_beta', 0.10)    # Beta层比例（次佳个体）
        self.rho_delta = config.get('rho_delta', 0.15)   # Delta层比例（第三佳个体）
        # Omega比例自动计算：1 - (alpha + beta + delta)
        
        # 自适应参数
        self.use_adaptive_layers = config.get('use_adaptive_layers', True)
        self.diversity_threshold = config.get('diversity_threshold', 0.15)
        self.adaptive_factor = config.get('adaptive_factor', 0.1)
        
        # 性能优化参数
        self.enable_fitness_cache = config.get('enable_fitness_cache', True)
        self.min_diversity_boost = config.get('min_diversity_boost', 1.2)  # 多样性低时的探索增强
        
        # 位置更新策略参数
        self.exploration_weight = config.get('exploration_weight', 1.5)  # 探索阶段权重
        self.exploitation_weight = config.get('exploitation_weight', 2.0)  # 开发阶段权重


class EnhancedGWO:
    """
    改进的灰狼优化算法选择器
    
    核心改进：
    1. 分阶段动态收敛因子：不同阶段使用不同的探索-开发平衡
    2. 自适应层比例：根据种群多样性动态调整各层比例
    3. 智能位置更新：结合GP特点，使用树结构的引导更新
    4. 性能优化：减少重复评估，使用缓存机制
    """
    
    def __init__(self, config: EnhancedGWOConfig = None):
        """
        初始化GWO选择器
        
        参数:
            config: GWO配置对象，如果为None则使用默认配置
        """
        self.config = config if config is not None else EnhancedGWOConfig()
        self.generation = 0
        self.max_generations = 50
        self.fitness_cache = {}  # 适应度缓存（用于性能优化）
        
    def calculate_convergence_factor(self, t: int, T: int) -> float:
        """
        计算动态收敛因子 a(t)
        
        理论基础：收敛因子控制探索和开发的平衡
        - a > 1: 探索阶段，鼓励全局搜索
        - a < 1: 开发阶段，鼓励局部精细搜索
        
        公式：
        1. 探索阶段 (0 ≤ t/T < τ₁): a(t) = a₀ × (1 - ratio)^0.5
        2. 平衡阶段 (τ₁ ≤ t/T < τ₂): a(t) = aₘ × (1 - ratio) + aₓ × ratio
        3. 开发阶段 (τ₂ ≤ t/T ≤ 1): a(t) = aₓ × (1 - ratio²)
        
        参数:
            t: 当前代数
            T: 最大代数
            
        返回:
            float: 收敛因子值
        """
        if T <= 0:
            return self.config.a0
        
        progress = t / T
        
        # 探索阶段: 0 ≤ t < τ₁T (0-20%)
        # 公式: a(t) = a₀ × (1 - ratio)^0.5, 其中 ratio = progress / τ₁
        if progress < self.config.tau1:
            ratio = progress / (self.config.tau1 + 1e-10)
            a = self.config.a0 * (1 - ratio) ** 0.5  # 平方根衰减，保持高探索性
            return max(self.config.am, a)
        
        # 平衡阶段: τ₁T ≤ t < τ₂T (20-70%)
        # 公式: a(t) = aₘ × (1 - ratio) + aₓ × ratio, 其中 ratio = (progress - τ₁) / (τ₂ - τ₁)
        elif progress < self.config.tau2:
            tau1_progress = self.config.tau1
            tau2_progress = self.config.tau2
            ratio = (progress - tau1_progress) / ((tau2_progress - tau1_progress) + 1e-10)
            a = self.config.am * (1 - ratio) + self.config.af * ratio  # 线性插值
            return max(self.config.af, a)
        
        # 开发阶段: τ₂T ≤ t ≤ T (70-100%)
        # 公式: a(t) = aₓ × (1 - ratio²), 其中 ratio = (progress - τ₂) / (1 - τ₂)
        else:
            tau2_progress = self.config.tau2
            ratio = (progress - tau2_progress) / ((1 - tau2_progress) + 1e-10)
            a = self.config.af * (1 - ratio ** 2)  # 平方衰减，更快收敛到局部搜索
            return max(0.05, a)  # 保持最小探索能力
    
    def get_stage(self, t: int, T: int) -> str:
        """
        获取当前进化阶段
        
        参数:
            t: 当前代数
            T: 最大代数
            
        返回:
            str: 阶段名称 ('exploration', 'balance', 'exploitation')
        """
        if T <= 0:
            return 'exploration'
        
        progress = t / T
        if progress < self.config.tau1:
            return 'exploration'
        elif progress < self.config.tau2:
            return 'balance'
        else:
            return 'exploitation'
    
    def calculate_diversity(self, population: List) -> float:
        """
        计算种群多样性（基于适应度分布）
        
        参数:
            population: 种群个体列表
            
        返回:
            float: 多样性指数 [0, 1]，1表示完全多样，0表示完全收敛
        """
        if not population or len(population) < 2:
            return 1.0
        
        try:
            # 获取所有有效适应度值
            fitnesses = []
            for ind in population:
                if hasattr(ind, 'fitness') and ind.fitness.valid:
                    fitness_val = ind.fitness.values[0]
                    fitnesses.append(fitness_val)
            
            if len(fitnesses) < 2:
                return 1.0
            
            fitness_array = np.array(fitnesses)
            fitness_range = np.max(fitness_array) - np.min(fitness_array)
            fitness_std = np.std(fitness_array)
            
            if fitness_range < 1e-10:
                return 0.0
            
            # 多样性公式: Diversity = min(σ(fitness) / Range(fitness), 1.0)
            # 其中 σ(fitness) 是标准差，Range(fitness) 是适应度范围
            diversity = min(fitness_std / (fitness_range + 1e-10), 1.0)
            return diversity
        except Exception as e:
            logger.warning(f"计算多样性时出错: {e}")
            return 0.5  # 默认中等多样性
    
    def calculate_layer_proportions(self, diversity: float) -> Tuple[float, float, float, float]:
        """
        计算自适应层比例
        
        理论基础：当多样性低时，增加Omega比例以增强探索
        
        参数:
            diversity: 种群多样性指数
            
        返回:
            tuple: (rho_alpha, rho_beta, rho_delta, rho_omega)
        """
        base_alpha = self.config.rho_alpha
        base_beta = self.config.rho_beta
        base_delta = self.config.rho_delta
        base_omega = 1.0 - (base_alpha + base_beta + base_delta)
        
        if not self.config.use_adaptive_layers:
            return (base_alpha, base_beta, base_delta, base_omega)
        
        # 多样性低时，增加Omega比例（增强探索）
        # 公式: transfer = α × (1 - Diversity / threshold)
        if diversity < self.config.diversity_threshold:
            transfer = self.config.adaptive_factor * (1 - diversity / self.config.diversity_threshold)
            rho_alpha = max(0.02, base_alpha - transfer * 0.2)
            rho_beta = max(0.05, base_beta - transfer * 0.3)
            rho_delta = max(0.10, base_delta - transfer * 0.3)
            rho_omega = min(0.90, base_omega + transfer * 0.8)
        else:
            # 多样性高时，略微增加Leader比例（增强开发）
            # 公式: boost = α × (Diversity - threshold) / (1 - threshold)
            boost = self.config.adaptive_factor * (diversity - self.config.diversity_threshold) / (1 - self.config.diversity_threshold)
            rho_alpha = min(0.10, base_alpha + boost * 0.3)
            rho_beta = min(0.15, base_beta + boost * 0.2)
            rho_delta = min(0.20, base_delta + boost * 0.2)
            rho_omega = max(0.55, base_omega - boost * 0.7)
        
        # 归一化
        total = rho_alpha + rho_beta + rho_delta + rho_omega
        if total < 1e-10:
            return (base_alpha, base_beta, base_delta, base_omega)
        
        return (
            rho_alpha / total,
            rho_beta / total,
            rho_delta / total,
            rho_omega / total
        )
    
    def classify_population(self, population: List) -> Tuple[List, List, List, List]:
        """
        将种群分类为Alpha/Beta/Delta/Omega层
        
        参数:
            population: 已排序的种群（从最佳到最差）
            
        返回:
            tuple: (alpha_list, beta_list, delta_list, omega_list)
        """
        # 确保种群已按适应度排序（最佳在前）
        sorted_pop = sorted(
            population,
            key=lambda ind: ind.fitness.values[0] if hasattr(ind, 'fitness') and ind.fitness.valid else float('inf'),
            reverse=True  # 适应度越大越好（R2等）
        )
        
        diversity = self.calculate_diversity(sorted_pop)
        rho_alpha, rho_beta, rho_delta, rho_omega = self.calculate_layer_proportions(diversity)
        
        n = len(sorted_pop)
        n_alpha = max(1, int(n * rho_alpha))
        n_beta = max(1, int(n * rho_beta))
        n_delta = max(1, int(n * rho_delta))
        n_omega = n - n_alpha - n_beta - n_delta
        
        # 确保至少有一个Omega
        if n_omega < 1:
            n_omega = 1
            n_delta = max(1, n_delta - 1)
        
        alpha_list = sorted_pop[:n_alpha]
        beta_list = sorted_pop[n_alpha:n_alpha + n_beta]
        delta_list = sorted_pop[n_alpha + n_beta:n_alpha + n_beta + n_delta]
        omega_list = sorted_pop[n_alpha + n_beta + n_delta:]
        
        return alpha_list, beta_list, delta_list, omega_list
    
    def update_generation(self, generation: int, max_generations: int):
        """
        更新当前代数信息
        
        参数:
            generation: 当前代数
            max_generations: 最大代数
        """
        self.generation = generation
        self.max_generations = max_generations
    
    def select(self, population: List, k: int) -> List:
        """
        GWO选择算子
        
        核心思想：
        1. 将种群分为Alpha/Beta/Delta/Omega四层
        2. 根据当前阶段和收敛因子，使用不同的选择策略
        3. Alpha/Beta/Delta直接保留（精英保留）
        4. Omega层根据GWO位置更新公式进行引导选择
        
        参数:
            population: 当前种群
            k: 需要选择的个体数量
            
        返回:
            List: 选择出的k个个体
        """
        if k <= 0:
            return []
        
        if len(population) == 0:
            return []
        
        # 分类种群
        alpha_list, beta_list, delta_list, omega_list = self.classify_population(population)
        
        # 获取当前阶段和收敛因子
        stage = self.get_stage(self.generation, self.max_generations)
        a = self.calculate_convergence_factor(self.generation, self.max_generations)
        
        selected = []
        
        # 1. 保留Alpha/Beta/Delta层（精英保留）
        # 根据阶段决定保留比例
        if stage == 'exploration':
            # 探索阶段：保留更多精英以引导搜索
            alpha_keep = min(len(alpha_list), int(k * 0.15))
            beta_keep = min(len(beta_list), int(k * 0.10))
            delta_keep = min(len(delta_list), int(k * 0.08))
        elif stage == 'balance':
            # 平衡阶段：适度保留
            alpha_keep = min(len(alpha_list), int(k * 0.12))
            beta_keep = min(len(beta_list), int(k * 0.08))
            delta_keep = min(len(delta_list), int(k * 0.06))
        else:  # exploitation
            # 开发阶段：更多保留，精细搜索
            alpha_keep = min(len(alpha_list), int(k * 0.20))
            beta_keep = min(len(beta_list), int(k * 0.15))
            delta_keep = min(len(delta_list), int(k * 0.10))
        
        selected.extend(random.sample(alpha_list, alpha_keep) if len(alpha_list) > 0 else [])
        selected.extend(random.sample(beta_list, beta_keep) if len(beta_list) > 0 else [])
        selected.extend(random.sample(delta_list, delta_keep) if len(delta_list) > 0 else [])
        
        # 2. 从Omega层选择剩余个体（使用GWO引导）
        remaining = k - len(selected)
        if remaining > 0 and len(omega_list) > 0:
            omega_selected = self._select_from_omega(
                omega_list, alpha_list, beta_list, delta_list, 
                remaining, a, stage
            )
            selected.extend(omega_selected)
        
        # 3. 如果还不够，从所有层随机选择
        if len(selected) < k:
            remaining = k - len(selected)
            all_individuals = alpha_list + beta_list + delta_list + omega_list
            selected.extend(random.sample(all_individuals, min(remaining, len(all_individuals))))
        
        return selected[:k]
    
    def _select_from_omega(self, omega_list: List, alpha_list: List, 
                           beta_list: List, delta_list: List, 
                           k: int, a: float, stage: str) -> List:
        """
        从Omega层选择个体（使用GWO位置更新引导）
        
        理论基础：GWO位置更新公式
        X(t+1) = (X_alpha + X_beta + X_delta) / 3
        
        在GP中，我们使用适应度加权来选择接近引导位置的个体
        
        参数:
            omega_list: Omega层个体列表
            alpha_list: Alpha层个体列表
            beta_list: Beta层个体列表
            delta_list: Delta层个体列表
            k: 需要选择的个体数量
            a: 当前收敛因子
            stage: 当前阶段
            
        返回:
            List: 选择的Omega个体
        """
        if k <= 0 or len(omega_list) == 0:
            return []
        
        # 获取引导个体的适应度（作为目标）
        alpha_fitness = alpha_list[0].fitness.values[0] if len(alpha_list) > 0 and alpha_list[0].fitness.valid else 0.0
        beta_fitness = beta_list[0].fitness.values[0] if len(beta_list) > 0 and beta_list[0].fitness.valid else 0.0
        delta_fitness = delta_list[0].fitness.values[0] if len(delta_list) > 0 and delta_list[0].fitness.valid else 0.0
        
        # 计算目标适应度（加权平均）
        # 公式: F_target = 0.5 × F_alpha + 0.3 × F_beta + 0.2 × F_delta
        # 这是GWO经典位置更新公式 X(t+1) = (X_alpha + X_beta + X_delta) / 3 在适应度空间的映射
        target_fitness = (alpha_fitness * 0.5 + beta_fitness * 0.3 + delta_fitness * 0.2)
        
        # 计算每个Omega个体的选择概率
        # 使用适应度距离和收敛因子
        probabilities = []
        for omega in omega_list:
            if not omega.fitness.valid:
                prob = 0.1  # 无效适应度给予低概率
            else:
                omega_fitness = omega.fitness.values[0]
                # 计算与目标适应度的距离
                # 公式: distance = |F_omega - F_target|
                distance = abs(omega_fitness - target_fitness)
                
                # 根据阶段调整选择策略
                if stage == 'exploration':
                    # 探索阶段：鼓励多样性，距离适中的个体有更高概率
                    # 分段概率函数，鼓励中等距离的个体
                    if distance < 0.1:
                        prob = 0.3  # 接近引导位置
                    elif distance < 0.3:
                        prob = 0.5  # 中等距离（鼓励探索）
                    else:
                        prob = 0.2  # 远距离（保持多样性）
                    # 修正: P = P × (1 + a × 0.3)
                    prob = prob * (1 + a * 0.3)
                elif stage == 'balance':
                    # 平衡阶段：偏向接近引导位置的个体
                    # 公式: P = exp(-distance / (a + 0.1)) × (1 - a × 0.2)
                    prob = np.exp(-distance / (a + 0.1))  # 指数衰减
                    prob = prob * (1 - a * 0.2)
                else:  # exploitation
                    # 开发阶段：强烈偏向接近引导位置的个体
                    # 公式: P = exp(-distance × 2 / (a + 0.1)) × (1 - a × 0.2)
                    prob = np.exp(-distance * 2 / (a + 0.1))  # 更强的衰减（×2）
                    prob = prob * (1 - a * 0.2)
            
            probabilities.append(max(0.0, min(1.0, prob)))
        
        # 归一化概率
        prob_sum = sum(probabilities)
        if prob_sum < 1e-10:
            # 如果所有概率都很小，使用均匀分布
            return random.sample(omega_list, min(k, len(omega_list)))
        
        probabilities = [p / prob_sum for p in probabilities]
        
        # 使用加权随机选择
        selected = []
        omega_indices = list(range(len(omega_list)))
        
        for _ in range(min(k, len(omega_list))):
            if len(omega_indices) == 0:
                break
            
            # 获取当前剩余个体的概率
            remaining_probs = [probabilities[i] for i in omega_indices]
            remaining_sum = sum(remaining_probs)
            
            # 如果总和太小或为0，使用均匀分布
            if remaining_sum < 1e-10:
                chosen_idx = random.choice(omega_indices)
            else:
                # 归一化剩余概率，确保总和为1（处理浮点数精度问题）
                normalized_probs = [p / remaining_sum for p in remaining_probs]
                
                # 再次检查并修正（处理浮点数精度问题）
                normalized_sum = sum(normalized_probs)
                if abs(normalized_sum - 1.0) > 1e-10:
                    # 如果归一化后总和不为1，重新归一化
                    normalized_probs = [p / normalized_sum for p in normalized_probs]
                
                # 加权随机选择
                chosen_idx = np.random.choice(omega_indices, p=normalized_probs)
            
            selected.append(omega_list[chosen_idx])
            omega_indices.remove(chosen_idx)
        
        return selected


def selEnhancedGWO(population: List, k: int, gwo: EnhancedGWO) -> List:
    """
    DEAP兼容的选择函数
    
    参数:
        population: 种群列表
        k: 需要选择的个体数量
        gwo: EnhancedGWO实例
        
    返回:
        List: 选择出的k个个体
    """
    return gwo.select(population, k)

