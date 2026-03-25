"""
变异策略类
"""


class MutationStrategy:
    """变异策略配置类"""
    
    def __init__(self, config: dict):
        """
        初始化变异策略配置
        
        参数:
            config: 配置字典，包含以下字段：
                - mut_prob: float, 变异概率（0.0-1.0）
                - strategy: str, 变异策略名称（如 "uniform"）
                - mutate_gen_full_min: int, 生成型突变最小高度（DEAP标准）
                - mutate_gen_full_max: int, 生成型突变最大高度（DEAP标准）
                - max_tree_height: int, 最大树高度限制（DEAP标准：叶子节点height=0）
        """
        self.mut_prob = config.get("mut_prob", 0.19)
        self.strategy = config.get("strategy", "uniform")
        self.mutate_gen_full_min = config.get("mutate_gen_full_min", 0)
        self.mutate_gen_full_max = config.get("mutate_gen_full_max", 2)
        self.max_tree_height = config.get("max_tree_height", config.get("max_tree_depth", 2))  # 兼容旧配置
    
    def get_config(self) -> dict:
        """
        获取变异策略配置
        
        返回:
            dict: 包含变异策略配置的字典
        """
        return {
            "mut_prob": self.mut_prob,
            "strategy": self.strategy,
            "mutate_gen_full_min": self.mutate_gen_full_min,
            "mutate_gen_full_max": self.mutate_gen_full_max,
            "max_tree_height": self.max_tree_height
        }
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置的有效性
        
        返回:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if not 0.0 <= self.mut_prob <= 1.0:
            return False, f"mut_prob必须在0.0-1.0之间，当前为{self.mut_prob}"
        
        if self.mutate_gen_full_min < 0 or self.mutate_gen_full_max < self.mutate_gen_full_min:
            return False, f"生成型突变深度参数无效: min={self.mutate_gen_full_min}, max={self.mutate_gen_full_max}"
        
        if self.max_tree_height < 0:
            return False, f"max_tree_height必须大于等于0，当前为{self.max_tree_height}"
        
        return True, ""

