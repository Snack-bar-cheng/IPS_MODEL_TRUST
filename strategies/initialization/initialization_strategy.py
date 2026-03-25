"""
初始化策略类
支持random和llm两种初始化方式
"""


class InitializationStrategy:
    """初始化策略配置类"""
    
    def __init__(self, config: dict):
        """
        初始化策略配置
        
        参数:
            config: 配置字典，包含以下字段：
                - random: dict, 随机初始化配置
                    - enabled: bool, 是否启用
                    - ratio: float, 占比（0.0-1.0）
                    - initial_min_depth: int, 最小深度
                    - initial_max_depth: int, 最大深度
                - llm: dict, LLM初始化配置
                    - enabled: bool, 是否启用
                    - ratio: float, 占比（0.0-1.0）
                    - llm_max_tree_depth: int, LLM特征的最大深度限制
                    - llm_feature_paths: list[str], LLM特征文件路径列表
        """
        self.random_config = config.get("random", {})
        self.llm_config = config.get("llm", {})
    
    def _get_llm_feature_paths(self) -> list:
        """
        获取LLM特征文件路径列表
        兼容 llm_feature_path (单数) 和 llm_feature_paths (复数) 两种配置方式
        """
        # 优先使用复数形式
        if "llm_feature_paths" in self.llm_config:
            paths = self.llm_config["llm_feature_paths"]
            # 确保返回列表
            return paths if isinstance(paths, list) else [paths]
        # 兼容单数形式
        elif "llm_feature_path" in self.llm_config:
            path = self.llm_config["llm_feature_path"]
            # 确保返回列表
            return path if isinstance(path, list) else [path]
        else:
            return []
    
    def get_config(self) -> dict:
        """
        获取初始化策略配置
        
        返回:
            dict: 包含初始化策略配置的字典
        """
        return {
            "random": {
                "enabled": self.random_config.get("enabled", True),
                "ratio": self.random_config.get("ratio", 1.0),
                "initial_min_height": self.random_config.get("initial_min_height", self.random_config.get("initial_min_depth", 1)),  # 兼容旧配置
                "initial_max_height": self.random_config.get("initial_max_height", self.random_config.get("initial_max_depth", 2))  # 兼容旧配置
            },
            "llm": {
                "enabled": self.llm_config.get("enabled", False),
                "ratio": self.llm_config.get("ratio", 0.0),
                "llm_max_tree_height": self.llm_config.get("llm_max_tree_height", self.llm_config.get("llm_max_tree_depth", 2)),  # 兼容旧配置
                "llm_feature_paths": self._get_llm_feature_paths()
            }
        }
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置的有效性
        
        返回:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        # 检查至少启用一种初始化方式
        random_enabled = self.random_config.get("enabled", True)
        llm_enabled = self.llm_config.get("enabled", False)
        
        if not random_enabled and not llm_enabled:
            return False, "至少需要启用一种初始化方式（random或llm）"
        
        # 检查比例总和
        random_ratio = self.random_config.get("ratio", 1.0) if random_enabled else 0.0
        llm_ratio = self.llm_config.get("ratio", 0.0) if llm_enabled else 0.0
        
        if random_enabled and llm_enabled:
            total_ratio = random_ratio + llm_ratio
            if abs(total_ratio - 1.0) > 0.01:  # 允许小的浮点误差
                return False, f"random和llm的比例之和应为1.0，当前为{total_ratio}"
        
        # 检查高度参数（兼容旧配置中的depth）
        if random_enabled:
            min_height = self.random_config.get("initial_min_height", self.random_config.get("initial_min_depth", 1))
            max_height = self.random_config.get("initial_max_height", self.random_config.get("initial_max_depth", 2))
            if min_height < 0 or max_height < min_height:
                return False, f"随机初始化高度参数无效: min_height={min_height}, max_height={max_height}"
        
        if llm_enabled:
            llm_max_height = self.llm_config.get("llm_max_tree_height", self.llm_config.get("llm_max_tree_depth", 2))
            if llm_max_height < 0:
                return False, f"LLM最大高度参数无效: {llm_max_height}"
            
            # 只有当llm_ratio > 0时才需要提供路径
            if llm_ratio > 0:
                llm_paths = self._get_llm_feature_paths()
                if not llm_paths:
                    return False, "启用LLM初始化且ratio > 0时必须提供llm_feature_paths或llm_feature_path"
        
        return True, ""

