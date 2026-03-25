"""
选择策略类
"""


class SelectionStrategy:
    """选择策略配置类"""
    
    def __init__(self, config: dict):
        """
        初始化选择策略配置
        
        参数:
            config: 配置字典，包含以下字段：
                - strategy: str, 选择策略名称（如 "tournament" 或 "gwo"）
                - tournament_size: int, 锦标赛选择的大小（如果使用tournament策略）
                - hof_size: int, 名人堂大小
                - enable_gwo: bool, 是否启用GWO算法（如果为True，strategy会被忽略）
                - gwo_type: str, GWO算法类型，"traditional"或"enhanced"（默认"enhanced"）
                - gwo_config: dict, GWO算法配置（可选，仅在enhanced类型时生效）
        """
        self.enable_gwo = config.get("enable_gwo", False)
        
        if self.enable_gwo:
            self.strategy = "gwo"
        else:
            self.strategy = config.get("strategy", "tournament")
        
        self.tournament_size = config.get("tournament_size", 7)
        self.hof_size = config.get("hof_size", 20)
        
        # GWO配置
        self.gwo_type = config.get("gwo_type", "enhanced")  # "traditional" 或 "enhanced"
        self.gwo_config = config.get("gwo_config", {})
    
    def get_config(self) -> dict:
        """
        获取选择策略配置
        
        返回:
            dict: 包含选择策略配置的字典
        """
        config_dict = {
            "strategy": self.strategy,
            "tournament_size": self.tournament_size,
            "hof_size": self.hof_size,
            "enable_gwo": self.enable_gwo
        }
        
        if self.enable_gwo:
            config_dict["gwo_type"] = self.gwo_type
            if self.gwo_type == "enhanced":
                config_dict["gwo_config"] = self.gwo_config
        
        return config_dict
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置的有效性
        
        返回:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if not self.enable_gwo and self.tournament_size < 1:
            return False, f"tournament_size必须大于0，当前为{self.tournament_size}"
        
        if self.hof_size < 1:
            return False, f"hof_size必须大于0，当前为{self.hof_size}"
        
        return True, ""

