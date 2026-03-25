"""
JSON格式化工具
用于格式化baseline结果JSON，使其更易读
"""

import json


def format_summary_paired_fields(summary_dict, indent_level, indent_str):
    """
    将 cross_validation 对象按配对字段格式化（mean 和 std 配对在同一行）
    
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


def format_json_compact(data, indent=2):
    """
    格式化JSON，对baseline结果使用紧凑格式
    
    使用递归格式化函数，对以下字段使用紧凑格式（同一行显示）：
    - feature_names: 数组在一行
    - train_metrics: 对象在一行
    - test_metrics: 对象在一行
    - cross_validation: mean 和 std 配对在同一行
    
    参数:
        data: 要格式化的数据
        indent: 缩进级别
    
    返回:
        格式化后的JSON字符串
    """
    def format_value(value, indent_level, indent_str):
        """递归格式化值"""
        if isinstance(value, dict):
            output_items = []
            current_indent = indent_str * indent_level
            
            for key, val in value.items():
                key_str = json.dumps(key, ensure_ascii=False)
                
                # 对 feature_names 使用紧凑格式（数组在一行）
                if key == "feature_names" and isinstance(val, list):
                    compact_array = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_array}')
                # 对 train_metrics 使用紧凑格式（对象在一行）
                elif key == "train_metrics" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 test_metrics 使用紧凑格式（对象在一行）
                elif key == "test_metrics" and isinstance(val, dict):
                    compact_dict = json.dumps(val, ensure_ascii=False, separators=(', ', ': '))
                    output_items.append(f'{current_indent}{key_str}: {compact_dict}')
                # 对 cross_validation 使用配对字段格式
                elif key == "cross_validation" and isinstance(val, dict):
                    cv_formatted = format_summary_paired_fields(val, indent_level + 1, indent_str)
                    output_items.append(f'{current_indent}{key_str}: {cv_formatted}')
                # 对 feature_importances 使用紧凑格式（每个对象在一行）
                elif key == "feature_importances":
                    if val is None:
                        output_items.append(f'{current_indent}{key_str}: null')
                    elif isinstance(val, list):
                        items = []
                        next_indent = indent_str * (indent_level + 1)
                        for item in val:
                            if isinstance(item, dict) and 'importance' in item and 'feature_name' in item:
                                # 将 feature_importance 对象格式化为同一行
                                compact_dict = json.dumps(item, ensure_ascii=False, separators=(', ', ': '))
                                items.append(next_indent + compact_dict)
                            else:
                                item_str = format_value(item, indent_level + 1, indent_str)
                                items.append(next_indent + item_str)
                        if not items:
                            output_items.append(f'{current_indent}{key_str}: []')
                        else:
                            feature_imp_str = '[\n' + ',\n'.join(items) + '\n' + current_indent + ']'
                            output_items.append(f'{current_indent}{key_str}: {feature_imp_str}')
                    else:
                        # 如果类型不对，使用默认格式化
                        val_str = format_value(val, indent_level + 1, indent_str)
                        output_items.append(f'{current_indent}{key_str}: {val_str}')
                else:
                    # 递归处理其他字段
                    val_str = format_value(val, indent_level + 1, indent_str)
                    output_items.append(f'{current_indent}{key_str}: {val_str}')
            
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
            
            for item in value:
                item_str = format_value(item, indent_level + 1, indent_str)
                items.append(next_indent + item_str)
            
            return '[\n' + ',\n'.join(items) + '\n' + current_indent + ']'
        
        else:
            # 基本类型直接序列化
            return json.dumps(value, ensure_ascii=False)
    
    indent_str = ' ' * indent
    return format_value(data, 0, indent_str)

