"""
KS检验工具函数
"""

from scipy import stats


def ks_test_for_distribution(X_train, X_test, feature_names, alpha=0.05):
    """
    使用KS检验检查训练集和测试集的分布是否一致
    
    参数:
        X_train: 训练集特征 (numpy数组)
        X_test: 测试集特征 (numpy数组)
        feature_names: 特征名称列表
        alpha: 显著性水平，默认0.05
    
    返回:
        is_consistent: 布尔值，True表示分布一致，False表示不一致
        p_values: 每个特征的p值列表
        failed_features: 未通过检验的特征列表
    """
    p_values = []
    is_consistent = True
    failed_features = []
    
    for i, feature_name in enumerate(feature_names):
        # 对每个特征进行KS检验
        statistic, p_value = stats.ks_2samp(X_train[:, i], X_test[:, i])
        p_values.append(p_value)
        
        # 如果p值小于显著性水平，认为分布不一致
        if p_value < alpha:
            is_consistent = False
            failed_features.append(feature_name)
    
    return is_consistent, p_values, failed_features

