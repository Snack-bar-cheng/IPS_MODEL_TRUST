"""
传统灰狼优化算法（Traditional Grey Wolf Optimizer）
基于Mirjalili等人2014年提出的原始GWO算法

核心特点：
1. 固定线性收敛因子：a(t) = 2 - 2t/T
2. 固定三层引导：Alpha/Beta/Delta（各1个）
3. 固定位置更新公式
4. 无自适应机制
"""

import random
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class TraditionalGWO:
    """
    传统灰狼优化算法选择器
    
    实现标准GWO算法：
    1. 线性收敛因子：a(t) = 2 - 2t/T
    2. 固定三层引导：Alpha（最优）、Beta（次优）、Delta（第三优）
    3. 位置更新：X(t+1) = (X_alpha + X_beta + X_delta) / 3
    """
    
    def __init__(self):
        """初始化传统GWO选择器"""
        self.generation = 0
        self.max_generations = 50
    
    def calculate_convergence_factor(self, t: int, T: int) -> float:
        """
        计算传统GWO的收敛因子（线性衰减）
        
        公式：a(t) = 2 - 2 × (t / T)
        
        参数:
            t: 当前代数
            T: 最大代数
            
        返回:
            float: 收敛因子值，从2线性衰减到0
        """
        if T <= 0:
            return 2.0
        
        a = 2.0 - 2.0 * (t / T)
        return max(0.0, a)  # 确保a >= 0
    
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
        传统GWO选择算子
        
        核心思想：
        1. 将种群分为Alpha/Beta/Delta三层（各1个）和Omega层（其余）
        2. 直接保留Alpha/Beta/Delta（精英保留）
        3. Omega层根据GWO位置更新公式进行引导选择
        
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
        
        # 确保种群已按适应度排序（最佳在前）
        sorted_pop = sorted(
            population,
            key=lambda ind: ind.fitness.values[0] if hasattr(ind, 'fitness') and ind.fitness.valid else float('inf'),
            reverse=True  # 适应度越大越好（R2等）
        )
        
        # 传统GWO：固定三层，每层1个
        if len(sorted_pop) < 3:
            # 如果种群小于3个，直接返回所有个体
            return sorted_pop[:k]
        
        alpha = sorted_pop[0]  # 最优个体
        beta = sorted_pop[1]   # 次优个体
        delta = sorted_pop[2]  # 第三优个体
        omega_list = sorted_pop[3:]  # 其余个体
        
        selected = []
        
        # 1. 保留Alpha/Beta/Delta（精英保留）
        # 传统GWO中，这三个个体总是被保留
        selected.append(alpha)
        if k > 1:
            selected.append(beta)
        if k > 2:
            selected.append(delta)
        
        # 2. 从Omega层选择剩余个体（使用传统GWO引导）
        remaining = k - len(selected)
        if remaining > 0 and len(omega_list) > 0:
            omega_selected = self._select_from_omega(
                omega_list, alpha, beta, delta, remaining
            )
            selected.extend(omega_selected)
        
        # 3. 如果还不够，从所有个体随机选择
        if len(selected) < k:
            remaining = k - len(selected)
            all_individuals = sorted_pop
            selected.extend(random.sample(all_individuals, min(remaining, len(all_individuals))))
        
        return selected[:k]
    
    def _select_from_omega(self, omega_list: List, alpha, beta, delta, k: int) -> List:
        """
        从Omega层选择个体（使用传统GWO位置更新引导）
        
        理论基础：传统GWO位置更新公式
        X(t+1) = (X_alpha + X_beta + X_delta) / 3
        
        在GP中，我们使用适应度加权来选择接近引导位置的个体
        
        参数:
            omega_list: Omega层个体列表
            alpha: Alpha个体（最优）
            beta: Beta个体（次优）
            delta: Delta个体（第三优）
            k: 需要选择的个体数量
            
        返回:
            List: 选择的Omega个体
        """
        if k <= 0 or len(omega_list) == 0:
            return []
        
        # 获取引导个体的适应度
        alpha_fitness = alpha.fitness.values[0] if hasattr(alpha, 'fitness') and alpha.fitness.valid else 0.0
        beta_fitness = beta.fitness.values[0] if hasattr(beta, 'fitness') and beta.fitness.valid else 0.0
        delta_fitness = delta.fitness.values[0] if hasattr(delta, 'fitness') and delta.fitness.valid else 0.0
        
        # 传统GWO：等权重平均（标准公式）
        # X(t+1) = (X_alpha + X_beta + X_delta) / 3
        target_fitness = (alpha_fitness + beta_fitness + delta_fitness) / 3.0
        
        # 计算当前收敛因子
        a = self.calculate_convergence_factor(self.generation, self.max_generations)
        
        # 计算每个Omega个体的选择概率
        # 使用指数衰减函数：P = exp(-distance / (a + 0.1))
        probabilities = []
        for omega in omega_list:
            if not omega.fitness.valid:
                prob = 0.1  # 无效适应度给予低概率
            else:
                omega_fitness = omega.fitness.values[0]
                # 计算与目标适应度的距离
                distance = abs(omega_fitness - target_fitness)
                
                # 传统GWO：使用简单的指数衰减
                # 当a较大时（探索阶段），概率分布更均匀
                # 当a较小时（开发阶段），更偏向接近引导位置的个体
                prob = np.exp(-distance / (a + 0.1))
            
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
                # 归一化剩余概率
                normalized_probs = [p / remaining_sum for p in remaining_probs]
                
                # 再次检查并修正（处理浮点数精度问题）
                normalized_sum = sum(normalized_probs)
                if abs(normalized_sum - 1.0) > 1e-10:
                    normalized_probs = [p / normalized_sum for p in normalized_probs]
                
                # 加权随机选择
                chosen_idx = np.random.choice(omega_indices, p=normalized_probs)
            
            selected.append(omega_list[chosen_idx])
            omega_indices.remove(chosen_idx)
        
        return selected


def selTraditionalGWO(population: List, k: int, gwo: TraditionalGWO) -> List:
    """
    DEAP兼容的选择函数
    
    参数:
        population: 种群列表
        k: 需要选择的个体数量
        gwo: TraditionalGWO实例
        
    返回:
        List: 选择出的k个个体
    """
    return gwo.select(population, k)

