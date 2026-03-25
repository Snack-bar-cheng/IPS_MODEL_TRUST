"""
灰狼优化算法（Grey Wolf Optimizer for Genetic Programming）
包含传统GWO和改进GWO两种实现
"""

from .enhanced_gwo import EnhancedGWO, EnhancedGWOConfig
from .traditional_gwo import TraditionalGWO

__all__ = ['EnhancedGWO', 'EnhancedGWOConfig', 'TraditionalGWO']

