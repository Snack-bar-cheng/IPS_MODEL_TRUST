"""
JSON工具函数
"""


def format_json_lists_to_single_line(json_str):
    """
    将JSON字符串中的列表格式化为一行显示
    
    参数:
        json_str: JSON字符串
    
    返回:
        formatted_json_str: 格式化后的JSON字符串
    """
    lines = json_str.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 检测列表开始：行以 [ 结尾（如 "key": [）
        if stripped.endswith('['):
            # 提取缩进和键名部分（保留原始格式）
            indent = len(line) - len(line.lstrip())
            key_part = line.rstrip().rstrip('[').rstrip()
            
            # 收集列表的所有行
            list_lines = []
            i += 1
            
            # 收集列表项直到遇到 ]
            while i < len(lines):
                current_line = lines[i]
                current_stripped = current_line.strip()
                
                # 检查是否是列表结束
                if current_stripped == ']' or current_stripped == '],':
                    # 列表结束，合并为一行
                    # 提取所有列表项（去除逗号）
                    list_items = []
                    for l in list_lines:
                        item = l.strip().rstrip(',')
                        if item:
                            list_items.append(item)
                    
                    # 格式化为一行
                    list_content = ', '.join(list_items)
                    # 处理结尾的逗号
                    ending = ',' if current_stripped == '],' else ''
                    result_lines.append(key_part + ' [' + list_content + ']' + ending)
                    i += 1
                    break
                else:
                    list_lines.append(current_line)
                    i += 1
        else:
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)
