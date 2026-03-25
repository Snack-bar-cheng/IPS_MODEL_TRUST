import json
import os

def count_features(json_file_path):
    """
    读取JSON文件并统计features字段中的元素个数
    
    Args:
        json_file_path: JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件 {json_file_path} 不存在")
        return
    
    # 检查文件是否为空
    file_size = os.path.getsize(json_file_path)
    if file_size == 0:
        print(f"错误: 文件 {json_file_path} 为空（0字节）")
        return
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"错误: 文件 {json_file_path} 内容为空")
                return
            data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return
    
    # 检查是否是字典
    if not isinstance(data, dict):
        print(f"错误: JSON文件的根元素不是字典，而是 {type(data).__name__}")
        return
    
    # 检查是否有features字段
    if "features" not in data:
        print(f"错误: JSON文件中没有找到 'features' 字段")
        return
    
    features = data["features"]
    
    # 检查features是否是list
    if not isinstance(features, list):
        print(f"错误: 'features' 字段不是list，而是 {type(features).__name__}")
        return
    
    # 统计features字段中的元素个数
    element_count = len(features)
    print(f"features字段中的元素个数: {element_count}")


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # JSON文件路径（与脚本在同一目录）
    json_file = os.path.join(script_dir, "llm_Ash_Deformation_20251129_113945.json")
    
    # 执行统计
    count_features(json_file)
