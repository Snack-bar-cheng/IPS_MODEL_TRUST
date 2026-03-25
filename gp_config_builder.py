"""
GP配置生成器
整合所有策略配置，生成完整的GP配置JSON
"""

from strategies import (
    InitializationStrategy,
    SelectionStrategy,
    CrossoverStrategy,
    MutationStrategy,
    FunctionSetPool,
    HighFunctionStrategy
)


class GPConfigBuilder:
    """GP配置生成器类"""
    
    def __init__(self, config: dict):
        """
        初始化GP配置生成器
        
        参数:
            config: 完整的GP配置字典，包含以下部分：
                - population_size: int, 种群大小
                - generations: int, 进化代数
                - elitism_prob: float, 精英保留比例
                - cv_folds: int, 交叉验证折数
                - scale_factor: float, 适应度缩放因子
                - initialization: dict, 初始化策略配置
                - selection: dict, 选择策略配置
                - crossover: dict, 交叉策略配置
                - mutation: dict, 变异策略配置
                - function_set: dict, 函数集配置
                - residual_fitting: dict, 残差拟合配置
        """
        self.population_size = config.get("population_size", 100)
        self.generations = config.get("generations", 50)
        self.elitism_prob = config.get("elitism_prob", 0.01)
        self.cv_folds = config.get("cv_folds", 5)
        self.scale_factor = config.get("scale_factor", 100)
        self.config = config  # 保存原始配置，以便在build_config中使用
        
        # 初始化各个策略类
        self.initialization = InitializationStrategy(config.get("initialization", {}))
        self.selection = SelectionStrategy(config.get("selection", {}))
        self.crossover = CrossoverStrategy(config.get("crossover", {}))
        self.mutation = MutationStrategy(config.get("mutation", {}))
        self.function_set = FunctionSetPool(config.get("function_set", {}))
        self.high_function = HighFunctionStrategy(config.get("high_function", {}))
    
    def build_config(self) -> dict:
        """
        构建完整的GP配置JSON
        
        返回:
            dict: 完整的GP配置字典
        """
        # 验证所有策略配置
        validation_results = []
        validation_results.append(("initialization", self.initialization.validate()))
        validation_results.append(("selection", self.selection.validate()))
        validation_results.append(("crossover", self.crossover.validate()))
        validation_results.append(("mutation", self.mutation.validate()))
        validation_results.append(("function_set", self.function_set.validate()))
        validation_results.append(("high_function", self.high_function.validate()))
        
        # 检查验证结果
        for strategy_name, (is_valid, error_msg) in validation_results:
            if not is_valid:
                raise ValueError(f"{strategy_name}策略配置无效: {error_msg}")
        
        # 验证基本参数
        if self.population_size < 1:
            raise ValueError(f"population_size必须大于0，当前为{self.population_size}")
        if self.generations < 1:
            raise ValueError(f"generations必须大于0，当前为{self.generations}")
        if not 0.0 <= self.elitism_prob <= 1.0:
            raise ValueError(f"elitism_prob必须在0.0-1.0之间，当前为{self.elitism_prob}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds必须大于等于2，当前为{self.cv_folds}")
        
        # 验证交叉和变异概率之和（允许小的误差，因为可能还有其他操作）
        cx_prob = self.crossover.cx_prob
        mut_prob = self.mutation.mut_prob
        total_prob = cx_prob + mut_prob + self.elitism_prob
        if total_prob > 1.0 + 0.01:  # 只检查是否超过1.0
            raise ValueError(f"cx_prob + mut_prob + elitism_prob 不应超过1.0，当前为{total_prob}")
        
        # SHAP配置兼容
        shap_cfg = self.config.get("shap", {})
        if "open" not in shap_cfg:
            shap_cfg["open"] = self.config.get("shap_open", False)
        shap_cfg.setdefault("background_sample_size", None)
        shap_cfg.setdefault("explain_sample_size", None)
        shap_cfg.setdefault("ks_pvalue_threshold", 0.05)
        shap_cfg.setdefault("max_attempts", 5)

        # 构建配置字典
        config_dict = {
            "population_size": self.population_size,
            "generations": self.generations,
            "elitism_prob": self.elitism_prob,
            "cv_folds": self.cv_folds,
            "scale_factor": self.scale_factor,
            "initialization": self.initialization.get_config(),
            "selection": self.selection.get_config(),
            "crossover": self.crossover.get_config(),
            "mutation": self.mutation.get_config(),
            "function_set": self.function_set.get_config(),
            "high_function": self.high_function.get_config(),
            "residual_fitting": self.config.get("residual_fitting", {}),  # 保留原始配置中的残差拟合设置
            "baseline": self.config.get("baseline", {}),  # 保留原始配置中的baseline设置
            "shap": shap_cfg,  # SHAP配置
        }
        
        # 保留原始配置中的全局参数（如果存在）
        # 这些参数不在策略类中处理，需要直接从原始配置中保留
        global_params = [
            "max_tree_height",      # 全局最大树高度限制
            "cv_random_state",      # 交叉验证随机种子
            "shap_open",            # 兼容旧配置的SHAP开关
            # 可以在这里添加其他全局参数
        ]
        for param in global_params:
            if param in self.config:
                config_dict[param] = self.config[param]
        
        return config_dict
    
    def get_config_json(self) -> str:
        """
        获取GP配置的JSON字符串
        
        返回:
            str: JSON格式的配置字符串
        """
        import json
        config_dict = self.build_config()
        return json.dumps(config_dict, indent=2, ensure_ascii=False)

