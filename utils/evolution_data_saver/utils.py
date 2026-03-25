"""
工具函数模块
包含JSON格式化、路径转换、类型转换等工具函数
"""

import json
import os
import numpy as np


def format_summary_paired_fields(summary_dict, indent_level, indent_str):
    """
    将 summary 对象按配对字段格式化（mean 和 std 配对在同一行）
    
    例如：
    {
      "r2_mean": 0.723,
      "r2_std": 0.010,
      "mse_mean": 26542,
      "mse_std": 1388
    }
    
    转换为：
    {
      "r2_mean": 0.723, "r2_std": 0.010,
      "mse_mean": 26542, "mse_std": 1388
    }
    
    参数:
        summary_dict: summary 字典
        indent_level: 缩进级别
        indent_str: 缩进字符串
    
    返回:
        格式化后的字符串
    """
    if not summary_dict:
        return '{}'
    
    current_indent = indent_str * indent_level
    
    # 分离 mean 和 std 字段，以及其他字段
    mean_fields = {}
    std_fields = {}
    other_fields = {}
    
    for key, value in summary_dict.items():
        if key.endswith('_mean'):
            base_key = key[:-5]  # 移除 '_mean'
            mean_fields[base_key] = (key, value)
        elif key.endswith('_std'):
            base_key = key[:-4]  # 移除 '_std'
            std_fields[base_key] = (key, value)
        else:
            other_fields[key] = value
    
    # 构建输出项
    output_items = []
    
    # 处理配对字段（mean 和 std）
    paired_keys = set(mean_fields.keys()) | set(std_fields.keys())
    for base_key in sorted(paired_keys):
        mean_key, mean_val = mean_fields.get(base_key, (None, None))
        std_key, std_val = std_fields.get(base_key, (None, None))
        
        if mean_key and std_key:
            # 两个字段都存在，配对在同一行
            mean_str = json.dumps(mean_key, ensure_ascii=False)
            mean_val_str = json.dumps(mean_val, ensure_ascii=False)
            std_str = json.dumps(std_key, ensure_ascii=False)
            std_val_str = json.dumps(std_val, ensure_ascii=False)
            paired_line = f'{current_indent}{mean_str}: {mean_val_str}, {std_str}: {std_val_str}'
            output_items.append(paired_line)
        elif mean_key:
            # 只有 mean
            mean_str = json.dumps(mean_key, ensure_ascii=False)
            mean_val_str = json.dumps(mean_val, ensure_ascii=False)
            output_items.append(f'{current_indent}{mean_str}: {mean_val_str}')
        elif std_key:
            # 只有 std
            std_str = json.dumps(std_key, ensure_ascii=False)
            std_val_str = json.dumps(std_val, ensure_ascii=False)
            output_items.append(f'{current_indent}{std_str}: {std_val_str}')
    
    # 处理其他字段
    for key, value in sorted(other_fields.items()):
        key_str = json.dumps(key, ensure_ascii=False)
        val_str = json.dumps(value, ensure_ascii=False)
        output_items.append(f'{current_indent}{key_str}: {val_str}')
    
    if not output_items:
        return '{}'
    
    return '{\n' + ',\n'.join(output_items) + '\n' + current_indent + '}'


def format_folds_by_field(folds_list, indent_level, indent_str):
    """
    将 folds 数组按字段分组格式化
    
    例如，将多个对象：
    [{"r2": 0.73, "mse": 24915}, {"r2": 0.72, "mse": 27014}]
    
    转换为数组，每个元素是一个对象，包含一个字段及其所有值：
    [
      {"r2": [0.73, 0.72]},
      {"mse": [24915, 27014]}
    ]
    
    参数:
        folds_list: folds 数组
        indent_level: 缩进级别
        indent_str: 缩进字符串
    
    返回:
        格式化后的字符串
    """
    if not folds_list:
        return '[]'
    
    # 获取所有字段名（使用第一个元素的键）
    field_names = list(folds_list[0].keys())
    
    # 按字段分组收集值
    field_groups = {}
    for field_name in field_names:
        field_groups[field_name] = [item.get(field_name) for item in folds_list]
    
    # 格式化每个字段为单独的对象
    current_indent = indent_str * indent_level
    next_indent = indent_str * (indent_level + 1)
    
    items = []
    for field_name in field_names:
        field_key = json.dumps(field_name, ensure_ascii=False)
        field_values = field_groups[field_name]
        # 将值数组格式化为紧凑格式
        values_str = json.dumps(field_values, ensure_ascii=False, separators=(', ', ': '))
        # 创建单个字段对象
        field_obj = f'{{{field_key}: {values_str}}}'
        items.append(next_indent + field_obj)
    
    return '[\n' + ',\n'.join(items) + '\n' + current_indent + ']'


def format_json_compact(data, indent=2):
    """
    格式化JSON
    
    使用递归格式化函数，对以下字段使用紧凑格式（同一行显示）：
    - feature_names: 数组在一行
    - feature_usage: 对象在一行
    - gp_hyperparameters: 对象在一行
    - fitness: 统计对象在一行
    - size_tree: 统计对象在一行
    - train_metrics: 对象在一行
    - test_metrics: 对象在一行
    - summary: 配对字段（mean 和 std）在同一行显示（如 r2_mean 和 r2_std）
    - 个体信息中的数值字段（fitness, mse, rmse, mae, size, depth）: 在同一行显示
    - folds 数组: 按字段分组显示（每个字段一行，值为数组）
    
    参数:
        data: 要格式化的数据
        indent: 缩进级别
    
    返回:
        格式化后的JSON字符串
    """
    def format_value(value, indent_level, indent_str):
        """递归格式化值"""
        if isinstance(value, dict):
            # 检测是否为个体信息字典（包含 rank 和 gp_expression）
            is_individual_info = "rank" in value and "gp_expression" in value
            
            # 定义数值字段列表（这些字段应该在同一行）
            numeric_fields = ["fitness", "mse", "rmse", "mae", "size", "height", "depth"]
            
            output_items = []
            numeric_parts = []  # 存储数值字段的键值对字符串
            current_indent = indent_str * indent_level
            
            ridge_formula_processed = False
            
            for key, val in value.items():
                key_str = json.dumps(key, ensure_ascii=False)
                
                # 对 feature_names 使用紧凑格式（数组在一行）
                if key == "feature_names" and isinstance(val, list):
                    compact_array = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_array}')
                # 对 feature_usage 使用紧凑格式（对象在一行）
                elif key == "feature_usage" and isinstance(val, dict):
                    # 如果ridge_formula已处理且有数值字段，先输出数值字段
                    if ridge_formula_processed and numeric_parts:
                        numeric_line = ', '.join(numeric_parts)
                        output_items.append(f'{current_indent}{numeric_line}')
                        numeric_parts = []
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 gp_hyperparameters 使用紧凑格式（对象在一行）
                elif key == "gp_hyperparameters" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 fitness 使用紧凑格式（对象在一行，如果是统计中的fitness）
                elif key == "fitness" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 size_tree 使用紧凑格式（对象在一行）
                elif key == "size_tree" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 train_metrics 使用紧凑格式（对象在一行）
                elif key == "train_metrics" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 test_metrics 使用紧凑格式（对象在一行）
                elif key == "test_metrics" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 summary 使用配对字段格式（mean 和 std 配对在同一行）
                elif key == "summary" and isinstance(val, dict):
                    summary_formatted = format_summary_paired_fields(val, indent_level + 1, indent_str)
                    output_items.append(f'{current_indent}{key_str}: {summary_formatted}')
                # 对 cross_validation 使用配对字段格式（mean 和 std 配对在同一行）
                elif key == "cross_validation" and isinstance(val, dict):
                    cross_validation_formatted = format_summary_paired_fields(val, indent_level + 1, indent_str)
                    output_items.append(f'{current_indent}{key_str}: {cross_validation_formatted}')
                # 对 cv_metrics 中的 folds 数组进行特殊处理（按字段分组）
                elif key == "folds" and isinstance(val, list) and len(val) > 0:
                    # 检查是否为 folds 数组（所有元素都是相同结构的字典）
                    if all(isinstance(item, dict) for item in val):
                        # 获取第一个元素的所有键
                        first_keys = list(val[0].keys()) if val else []
                        # 检查所有元素是否有相同的键结构
                        if all(set(item.keys()) == set(first_keys) for item in val):
                            # 按字段分组显示
                            folds_formatted = format_folds_by_field(val, indent_level + 1, indent_str)
                            output_items.append(f'{current_indent}{key_str}: {folds_formatted}')
                        else:
                            # 如果结构不一致，使用标准格式化
                            val_str = format_value(val, indent_level + 1, indent_str)
                            output_items.append(f'{current_indent}{key_str}: {val_str}')
                    else:
                        # 如果不是字典列表，使用标准格式化
                        val_str = format_value(val, indent_level + 1, indent_str)
                        output_items.append(f'{current_indent}{key_str}: {val_str}')
                # 如果是个体信息字典中的数值字段，收集起来
                elif is_individual_info and key in numeric_fields:
                    val_str = json.dumps(val, ensure_ascii=False)
                    numeric_parts.append(f'{key_str}: {val_str}')
                elif key == "ridge_formula":
                    # 处理ridge_formula
                    val_str = format_value(val, indent_level + 1, indent_str)
                    output_items.append(f'{current_indent}{key_str}: {val_str}')
                    ridge_formula_processed = True
                    # 如果有数值字段，立即输出
                    if numeric_parts:
                        numeric_line = ', '.join(numeric_parts)
                        output_items.append(f'{current_indent}{numeric_line}')
                        numeric_parts = []  # 清空，避免重复
                else:
                    # 递归处理其他字段
                    val_str = format_value(val, indent_level + 1, indent_str)
                    output_items.append(f'{current_indent}{key_str}: {val_str}')
            
            # 如果还有未输出的数值字段（可能没有ridge_formula），在最后输出
            if numeric_parts:
                numeric_line = ', '.join(numeric_parts)
                output_items.append(f'{current_indent}{numeric_line}')
            
            current_indent = indent_str * indent_level
            if not output_items:
                return '{}'
            return '{\n' + ',\n'.join(output_items) + '\n' + current_indent + '}'
        
        elif isinstance(value, list):
            if not value:
                return '[]'
            
            items = []
            current_indent = indent_str * indent_level
            next_indent = indent_str * (indent_level + 1)
            
            # 紧凑行格式的判断：支持旧格式(importance)与SHAP格式(mean_abs_shap/mean_shap)
            is_compact_feature_list = (
                len(value) > 0 and all(
                    isinstance(item, dict) and
                    'feature_name' in item and
                    (
                        'importance' in item or
                        'mean_abs_shap' in item or
                        'mean_shap' in item
                    )
                    for item in value
                )
            )
            
            if is_compact_feature_list:
                # 对特征重要性列表使用紧凑格式（每个对象单行）
                for item in value:
                    compact_item = json.dumps(item, ensure_ascii=False, separators=(', ', ': '))
                    items.append(next_indent + compact_item)
            else:
                # 其他数组使用标准格式化
                for item in value:
                    item_str = format_value(item, indent_level + 1, indent_str)
                    items.append(next_indent + item_str)
            
            return '[\n' + ',\n'.join(items) + '\n' + current_indent + ']'
        
        elif isinstance(value, range):
            # 将range对象转换为列表
            return json.dumps(list(value), ensure_ascii=False)
        else:
            # 基本类型直接序列化
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                # 如果无法序列化，尝试转换为字符串
                return json.dumps(str(value), ensure_ascii=False)
    
    indent_str = ' ' * indent
    return format_value(data, 0, indent_str)


def convert_to_relative_path(absolute_path, base_dir):
    """
    将绝对路径转换为相对于base_dir的相对路径
    
    参数:
        absolute_path: 绝对路径
        base_dir: 基准目录
    
    返回:
        相对路径字符串，如果转换失败则返回原始路径
    """
    try:
        if not absolute_path or not os.path.isabs(absolute_path):
            return absolute_path
        
        # 获取基准目录的绝对路径
        base_abs = os.path.abspath(base_dir)
        path_abs = os.path.abspath(absolute_path)
        
        # 计算相对路径
        rel_path = os.path.relpath(path_abs, base_abs)
        return rel_path
    except Exception:
        return absolute_path


def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

