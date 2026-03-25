"""
SHAP采样工具：从训练/测试集中抽取背景集与解释集，并做分布一致性检查。
"""

import numpy as np
from scipy.stats import ks_2samp
from typing import Optional, Tuple


def _ks_pass(sample: np.ndarray, full: np.ndarray, threshold: float) -> bool:
    """
    对每个特征进行KS检验，全部通过则返回True。
    """
    for i in range(full.shape[1]):
        p = ks_2samp(full[:, i], sample[:, i]).pvalue
        if np.isnan(p) or p < threshold:
            return False
    return True


def sample_with_ks(
    train: np.ndarray,
    test: np.ndarray,
    background_size: Optional[int],
    explain_size: Optional[int],
    ks_threshold: float = 0.05,
    max_attempts: int = 5,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    抽样背景集与解释集，并确保与原分布在KS检验下相似。
    - background_size / explain_size 为 None 时，使用完整数据。
    - 如果抽样多次仍未通过，回退使用完整数据。
    """
    rng = np.random.default_rng(random_state)
    train = np.array(train)
    test = np.array(test)

    def _try_sample(data: np.ndarray, size: Optional[int]) -> np.ndarray:
        if size is None or size >= len(data):
            return data
        idx = rng.choice(len(data), size=size, replace=False)
        return data[idx]

    # 背景采样
    bg = _try_sample(train, background_size)
    attempts = 0
    while attempts < max_attempts and not _ks_pass(bg, train, ks_threshold):
        bg = _try_sample(train, background_size)
        attempts += 1
    if attempts >= max_attempts and not _ks_pass(bg, train, ks_threshold):
        bg = train  # 回退
        print("[SHAP] 背景采样未通过KS检验，回退使用完整训练集。")

    # 解释采样
    ex = _try_sample(test, explain_size)
    attempts = 0
    while attempts < max_attempts and not _ks_pass(ex, test, ks_threshold):
        ex = _try_sample(test, explain_size)
        attempts += 1
    if attempts >= max_attempts and not _ks_pass(ex, test, ks_threshold):
        ex = test  # 回退
        print("[SHAP] 解释采样未通过KS检验，回退使用完整测试集。")

    return bg, ex


