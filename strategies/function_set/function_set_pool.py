"""
函数集注册池
管理GP中使用的所有原语（函数和终端）
"""


class FunctionSetPool:
    """函数集注册池配置类"""
    
    def __init__(self, config: dict):
        """
        初始化函数集配置
        
        参数:
            config: 配置字典，包含以下字段：
                - operators: list[str], 所有运算符列表（包括算术、选择、变换、激活等）
                - enable_ephemeral_constant: list[float, float] 或 False, 临时常数范围 [min, max]，False表示不启用
        """
        self.operators = config.get("operators", ["Add", "Sub", "Mul", "Div", "Max", "Min", "Mean", "Ln", "Log", "Squ", "Cub", "Sqrt", "Cbrt"])
        ephemeral_config = config.get("enable_ephemeral_constant", [0, 3000])
        if ephemeral_config is False:
            self.ephemeral_constant_range = None
        elif isinstance(ephemeral_config, list) and len(ephemeral_config) == 2:
            self.ephemeral_constant_range = ephemeral_config
        else:
            # 兼容旧配置：True 表示使用默认范围
            self.ephemeral_constant_range = [0, 3000] if ephemeral_config else None
    
    def get_config(self) -> dict:
        """
        获取函数集配置
        
        返回:
            dict: 包含函数集配置的字典
        """
        return {
            "operators": self.operators,
            "enable_ephemeral_constant": self.ephemeral_constant_range if self.ephemeral_constant_range is not None else False
        }
    
    def validate(self) -> tuple[bool, str]:
        """
        验证配置的有效性
        
        返回:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        if not isinstance(self.operators, list):
            return False, "operators必须是一个列表"
        
        if len(self.operators) == 0:
            return False, "operators列表不能为空"
        
        if self.ephemeral_constant_range is not None:
            if not isinstance(self.ephemeral_constant_range, list) or len(self.ephemeral_constant_range) != 2:
                return False, "enable_ephemeral_constant必须是一个包含两个元素的列表 [min, max] 或 False"
            min_val, max_val = self.ephemeral_constant_range
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                return False, "enable_ephemeral_constant的范围值必须是数字"
            if min_val >= max_val:
                return False, "enable_ephemeral_constant的最小值必须小于最大值"
        
        return True, ""

