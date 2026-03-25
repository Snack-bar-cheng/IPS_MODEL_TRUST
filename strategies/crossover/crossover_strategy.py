"""
交叉策略类
"""


class CrossoverStrategy:
    """交叉策略配置类"""
    
    def __init__(self, config: dict):
        """
        初始化交叉策略配置
        
        参数:
            config: 配置字典，包含以下字段：
                - cx_prob: float, 交叉概率（0.0-1.0）
                - strategy: str, 交叉策略名称（如 "one_point"）
                - max_tree_height: int, 最大树高度限制（DEAP标准：叶子节点height=0）
        """
        self.cx_prob = config.get("cx_prob", 0.8)
        self.strategy = config.get("strategy", "one_point")
        self.max_tree_height = config.get("max_tree_height", config.get("max_tree_depth", 2))  # 兼容旧配置
    
    def get_config(self) -> dict:
        """
        获取交叉策略配置
        
        返回:
            dict: 包含交叉策略配置的字典
        """
        return {
            "cx_prob": self.cx_prob,
            "strategy": self.strategy,
            "max_tree_height": self.max_tree_height
        }
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置的有效性
        
        返回:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if not 0.0 <= self.cx_prob <= 1.0:
            return False, f"cx_prob必须在0.0-1.0之间，当前为{self.cx_prob}"
        
        if self.max_tree_height < 0:
            return False, f"max_tree_height必须大于等于0，当前为{self.max_tree_height}"
        
        return True, ""

