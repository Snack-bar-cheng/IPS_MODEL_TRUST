"""
LLM响应解析模块
负责解析大模型返回的JSON响应
"""
import json
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def parse_llm_response(response_text: str) -> Optional[Dict]:
    """
    解析大模型返回的JSON响应（单个特征）
    
    参数:
        response_text: 模型返回的文本
    
    返回:
        解析后的字典，包含tree、description、notation，失败返回None
    """
    try:
        # 尝试提取JSON（可能包含markdown代码块）
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            # 验证必需字段
            if "tree" in result and "description" in result:
                return result
            else:
                logger.warning(f"JSON缺少必需字段，包含的字段: {list(result.keys())}")
                return None
        else:
            logger.warning(f"无法从响应中提取JSON: {response_text[:200]}")
            return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析失败: {e}, 响应前200字符: {response_text[:200]}")
        return None
    except Exception as e:
        logger.error(f"解析响应时出错: {e}")
        return None


def parse_batch_llm_response(response_text: str) -> Optional[List[Dict]]:
    """
    解析大模型返回的批量JSON响应（多个特征）
    
    参数:
        response_text: 模型返回的文本
    
    返回:
        解析后的特征列表，每个元素包含tree、description、notation，失败返回None
    """
    if not response_text or not response_text.strip():
        logger.warning("响应文本为空")
        return None
    
    try:
        # 方法1: 尝试直接解析（如果响应是纯JSON）
        try:
            result = json.loads(response_text.strip())
            if isinstance(result, list):
                # 验证每个元素的必需字段
                valid_features = []
                for i, feature in enumerate(result):
                    if isinstance(feature, dict) and "tree" in feature and "description" in feature:
                        valid_features.append(feature)
                    else:
                        logger.warning(f"第 {i+1} 个特征缺少必需字段: {list(feature.keys()) if isinstance(feature, dict) else type(feature)}")
                
                if valid_features:
                    return valid_features
        except json.JSONDecodeError:
            pass  # 继续尝试其他方法
        
        # 方法2: 尝试提取JSON数组（可能包含markdown代码块或其他文本）
        # 使用贪婪匹配找到最长的数组，然后尝试解析
        array_match = re.search(r'\[[\s\S]*\]', response_text, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            try:
                result = json.loads(json_str)
                
                # 验证是列表
                if not isinstance(result, list):
                    logger.warning(f"解析结果不是列表: {type(result)}")
                    return None
                
                # 验证每个元素的必需字段
                valid_features = []
                for i, feature in enumerate(result):
                    if isinstance(feature, dict) and "tree" in feature and "description" in feature:
                        valid_features.append(feature)
                    else:
                        logger.warning(f"第 {i+1} 个特征缺少必需字段: {list(feature.keys()) if isinstance(feature, dict) else type(feature)}")
                
                if valid_features:
                    return valid_features
            except json.JSONDecodeError as e:
                logger.warning(f"提取的JSON数组解析失败: {e}")
                # 尝试从截断的JSON中提取完整的特征对象
                try:
                    # 尝试提取所有完整的JSON对象（即使数组不完整）
                    valid_features = []
                    # 查找所有完整的 {...} 对象
                    object_pattern = r'\{\s*"tree"\s*:[\s\S]*?"description"\s*:[\s\S]*?\}'
                    object_matches = re.finditer(object_pattern, json_str, re.DOTALL)
                    for match in object_matches:
                        try:
                            obj_str = match.group(0)
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict) and "tree" in obj and "description" in obj:
                                valid_features.append(obj)
                        except json.JSONDecodeError:
                            continue
                    
                    if valid_features:
                        logger.info(f"从截断的JSON中提取到 {len(valid_features)} 个完整的特征对象")
                        return valid_features
                except Exception as parse_error:
                    logger.warning(f"尝试从截断JSON提取特征时出错: {parse_error}")
                
                # 尝试修复常见的JSON错误
                try:
                    # 尝试修复未闭合的字符串、数组等
                    json_str_fixed = _try_fix_json(json_str)
                    if json_str_fixed:
                        result = json.loads(json_str_fixed)
                        if isinstance(result, list):
                            valid_features = []
                            for i, feature in enumerate(result):
                                if isinstance(feature, dict) and "tree" in feature and "description" in feature:
                                    valid_features.append(feature)
                            if valid_features:
                                return valid_features
                except:
                    pass
        
        # 方法3: 尝试提取多个独立的JSON对象
        # 匹配 {...} 格式的对象
        object_matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        valid_features = []
        for match in object_matches:
            try:
                obj_str = match.group(0)
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "tree" in obj and "description" in obj:
                    valid_features.append(obj)
            except json.JSONDecodeError:
                continue
        
        if valid_features:
            logger.info(f"从响应中提取到 {len(valid_features)} 个有效特征")
            return valid_features
        
        logger.warning(f"无法从响应中提取有效的JSON: {response_text[:500]}")
        return None
        
    except Exception as e:
        logger.error(f"解析响应时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _try_fix_json(json_str: str) -> Optional[str]:
    """
    尝试修复常见的JSON格式错误
    
    参数:
        json_str: 可能有错误的JSON字符串
    
    返回:
        修复后的JSON字符串，如果无法修复返回None
    """
    try:
        # 尝试修复未闭合的字符串（在字符串末尾添加引号）
        # 这是一个简单的修复，可能不适用于所有情况
        fixed = json_str
        
        # 移除末尾的逗号（在数组或对象中）
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # 尝试闭合未闭合的数组或对象
        open_brackets = fixed.count('[') - fixed.count(']')
        open_braces = fixed.count('{') - fixed.count('}')
        
        if open_brackets > 0:
            fixed += ']' * open_brackets
        if open_braces > 0:
            fixed += '}' * open_braces
        
        # 验证修复后的JSON是否有效
        json.loads(fixed)
        return fixed
    except:
        return None


def tree_to_string(tree: Dict) -> str:
    """
    将tree结构转换为字符串表示（用于比较是否重复）
    
    参数:
        tree: tree字典
    
    返回:
        字符串表示
    """
    return json.dumps(tree, sort_keys=True, ensure_ascii=False)


def is_tree_duplicate(tree: Dict, existing_trees: List[Dict]) -> bool:
    """
    检查tree是否与已存在的tree重复
    
    参数:
        tree: 要检查的tree
        existing_trees: 已存在的tree列表
    
    返回:
        是否重复
    """
    tree_str = tree_to_string(tree)
    for existing_tree in existing_trees:
        if tree_to_string(existing_tree) == tree_str:
            return True
    return False

