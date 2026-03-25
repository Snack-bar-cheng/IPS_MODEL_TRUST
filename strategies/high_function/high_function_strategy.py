"""
High函数策略类
负责配置和管理High函数的注册和动态扩展
"""

from typing import Union, List


class HighFunctionStrategy:
    """High函数策略类"""
    
    def __init__(self, config: dict):
        """
        初始化High函数策略
        
        参数:
            config: High函数配置字典，包含：
                - enable: bool, 是否启用High函数
                - high_n: int or List[int], 静态模式下的High函数分支数
                - enable_dynamic_expansion: bool, 是否启用动态分支扩展
                - base_high_n: int, 动态扩展模式下的基础High函数分支数
                - expansion_interval: int, 动态扩展间隔（多少代扩展一次）
                - max_high_n: int, 动态扩展模式下的最大High函数分支数
                - expansion_llm_ratio: float, 动态扩展时新生成分支中LLM特征的比例（0-1之间）
                - open_dynamic: bool, 是否启用自适应扩展策略
                - growth_threshold: float, 精英平均fitness跨窗口平均增长阈值（R2*100尺度）
                - window_size: int, 自适应滑动窗口大小（可选，缺省使用expansion_interval）
        """
        self.enable = config.get("enable", False)
        self.high_n = config.get("high_n", 3)
        self.enable_dynamic_expansion = config.get("enable_dynamic_expansion", False)
        self.base_high_n = config.get("base_high_n", 3)
        self.expansion_interval = config.get("expansion_interval", 4)
        self.max_high_n = config.get("max_high_n", 10)
        self.expansion_llm_ratio = config.get("expansion_llm_ratio", 0.5)
        self.open_dynamic = config.get("open_dynamic", False)
        self.growth_threshold = config.get("growth_threshold", 1.0)
        self.window_size = config.get("window_size", None)
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置
        
        返回:
            tuple: (is_valid, error_message)
        """
        if not isinstance(self.enable, bool):
            return False, "enable必须是布尔值"
        
        if self.enable:
            # 如果启用动态扩展，检查动态扩展参数
            if self.enable_dynamic_expansion:
                if not isinstance(self.base_high_n, int) or self.base_high_n < 1:
                    return False, "base_high_n必须是大于0的整数"
                if not isinstance(self.expansion_interval, int) or self.expansion_interval < 1:
                    return False, "expansion_interval必须是大于0的整数"
                if not isinstance(self.max_high_n, int) or self.max_high_n < 1:
                    return False, "max_high_n必须是大于0的整数"
                if self.base_high_n > self.max_high_n:
                    return False, "base_high_n不能大于max_high_n"
                # 验证expansion_llm_ratio
                if not isinstance(self.expansion_llm_ratio, (int, float)):
                    return False, "expansion_llm_ratio必须是数字"
                if self.expansion_llm_ratio < 0 or self.expansion_llm_ratio > 1:
                    return False, "expansion_llm_ratio必须在0-1之间"
                if not isinstance(self.open_dynamic, bool):
                    return False, "open_dynamic必须是布尔值"
                if not isinstance(self.growth_threshold, (int, float)):
                    return False, "growth_threshold必须是数字"
                if self.growth_threshold <= 0:
                    return False, "growth_threshold必须大于0"
                if self.window_size is not None:
                    if not isinstance(self.window_size, int) or self.window_size < 2:
                        return False, "window_size必须是大于等于2的整数或None"
            else:
                # 如果未启用动态扩展，检查high_n参数
                if isinstance(self.high_n, int):
                    if self.high_n < 1:
                        return False, "high_n必须是大于0的整数"
                elif isinstance(self.high_n, list):
                    if not all(isinstance(n, int) and n > 0 for n in self.high_n):
                        return False, "high_n列表中的所有元素必须是大于0的整数"
                else:
                    return False, "high_n必须是整数或整数列表"
        
        return True, ""
    
    def get_config(self) -> dict:
        """
        获取配置字典
        
        返回:
            dict: High函数配置字典
        """
        return {
            "enable": self.enable,
            "high_n": self.high_n,
            "enable_dynamic_expansion": self.enable_dynamic_expansion,
            "base_high_n": self.base_high_n,
            "expansion_interval": self.expansion_interval,
            "max_high_n": self.max_high_n,
            "expansion_llm_ratio": self.expansion_llm_ratio,
            "open_dynamic": self.open_dynamic,
            "growth_threshold": self.growth_threshold,
            "window_size": self.window_size
        }

