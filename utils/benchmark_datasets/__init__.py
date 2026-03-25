"""
基准数据集加载模块
支持SRBench、UCI、Feynman等标准数据集
"""

from .dataset_loader import load_benchmark_dataset, list_available_datasets
from .srbench_loader import load_srbench_dataset
from .uci_loader import load_uci_dataset
from .feynman_loader import load_feynman_dataset

__all__ = [
    'load_benchmark_dataset',
    'list_available_datasets',
    'load_srbench_dataset',
    'load_uci_dataset',
    'load_feynman_dataset'
]

