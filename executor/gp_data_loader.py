"""
GP数据加载模块
"""

import pandas as pd
import random
from typing import Tuple, List


def load_data(data_config: dict, target_column: str) -> Tuple:
    """
    加载训练集和测试集数据
    
    参数:
        data_config: 数据配置字典
        target_column: 目标列名称
    
    返回:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    train_path = data_config['train_set_path']
    test_path = data_config['test_set_path']
    selected_features = data_config['selected_feature']
    
    # 加载CSV文件
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 提取特征和目标
    X_train = train_df[selected_features].values
    y_train = train_df[target_column].values
    X_test = test_df[selected_features].values
    y_test = test_df[target_column].values
    
    return X_train, y_train, X_test, y_test, selected_features


def initialize_population(toolbox, gp_config: dict, llm_features: List = None):
    """
    初始化种群
    
    参数:
        toolbox: 工具箱
        gp_config: GP配置
        llm_features: LLM特征列表
    
    返回:
        list: 初始种群
    """
    population_size = gp_config['population_size']
    init_config = gp_config['initialization']
    
    if init_config['llm']['enabled'] and llm_features and hasattr(toolbox, 'individual_llm'):
        llm_ratio = init_config['llm']['ratio']
        desired_llm_count = int(population_size * llm_ratio)
        actual_llm_count = min(desired_llm_count, len(llm_features))
        random_count = population_size - actual_llm_count
        
        pop_llm = [toolbox.individual_llm() for _ in range(actual_llm_count)]
        pop_random = [toolbox.individual_random() for _ in range(random_count)]
        pop = pop_llm + pop_random
        random.shuffle(pop)
    else:
        pop = [toolbox.individual_random() for _ in range(population_size)]
    
    return pop

