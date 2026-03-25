"""
特征组合策略模块
包含用于GP遗传编程的通用特征组合函数
用于将多个标量特征组合成向量特征
"""

import numpy as np


# ============================================================================
# 通用特征组合函数
# ============================================================================

def root_con(*args):
    """
    通用特征组合函数，将任意数量的标量组合成向量
    这是所有特征组合的核心函数，可以替代所有特定数量的组合函数
    
    注意：此函数只接受Float1（标量）类型的输入，不接受Vector1（数组）类型的输入
    这确保了High函数不能嵌套（High函数不能接受其他High函数的输出）
    
    Args:
        *args: 可变数量的输入标量（必须是标量，不能是数组）
    Returns:
        包含所有输入元素的numpy数组（1维数组）
        
    Examples:
        >>> root_con(1.0, 2.0)  # 两个特征组合
        array([1., 2.])
        >>> root_con(1.0, 2.0, 3.0)  # 三个特征组合
        array([1., 2., 3.])
        >>> root_con(1.0, 2.0, 3.0, 4.0, 5.0)  # 五个特征组合
        array([1., 2., 3., 4., 5.])
    """
    # 将所有参数转换为标量数组，只接受Float1（标量）类型
    arrays = []
    for i, arg in enumerate(args):
        arr = np.asarray(arg)
        
        # 检查输入类型：只接受标量（0维数组）
        if arr.ndim > 0:
            # 如果输入是多维数组（Vector1），抛出错误
            raise TypeError(
                f"root_con函数只接受Float1（标量）类型的输入，"
                f"但第{i+1}个参数是{arr.ndim}维数组（Vector1类型）。"
                f"High函数不能嵌套使用（不能接受其他High函数的输出）。"
            )
        
        # 如果是标量（0维），转换为1维数组
        arrays.append(arr.reshape(1))
    
    # 连接所有数组
    if arrays:
        feature_vector = np.concatenate(arrays, axis=0)
    else:
        feature_vector = np.array([])
    
    return feature_vector


# ============================================================================
# 特定数量的特征组合函数（为了兼容性保留）
# ============================================================================

def root_con1(v1):
    """
    一个特征组合（兼容性函数）
    Args:
        v1: 输入标量
    Returns:
        包含一个元素的numpy数组
    """
    return root_con(v1)


def root_con2(v1, v2):
    """
    两个特征组合（兼容性函数）
    Args:
        v1, v2: 输入标量
    Returns:
        包含两个元素的numpy数组
    """
    return root_con(v1, v2)


def root_con3(v1, v2, v3):
    """
    三个特征组合（兼容性函数）
    Args:
        v1, v2, v3: 输入标量
    Returns:
        包含三个元素的numpy数组
    """
    return root_con(v1, v2, v3)


def root_con4(v1, v2, v3, v4):
    """
    四个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4: 输入标量
    Returns:
        包含四个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4)


def root_con5(v1, v2, v3, v4, v5):
    """
    五个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4, v5: 输入标量
    Returns:
        包含五个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4, v5)


def root_con6(v1, v2, v3, v4, v5, v6):
    """
    六个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4, v5, v6: 输入标量
    Returns:
        包含六个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4, v5, v6)


def root_con7(v1, v2, v3, v4, v5, v6, v7):
    """
    七个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4, v5, v6, v7: 输入标量
    Returns:
        包含七个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4, v5, v6, v7)


def root_con8(v1, v2, v3, v4, v5, v6, v7, v8):
    """
    八个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4, v5, v6, v7, v8: 输入标量
    Returns:
        包含八个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4, v5, v6, v7, v8)


def root_con9(v1, v2, v3, v4, v5, v6, v7, v8, v9):
    """
    九个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4, v5, v6, v7, v8, v9: 输入标量
    Returns:
        包含九个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4, v5, v6, v7, v8, v9)


def root_con10(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10):
    """
    十个特征组合（兼容性函数）
    Args:
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10: 输入标量
    Returns:
        包含十个元素的numpy数组
    """
    return root_con(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)

