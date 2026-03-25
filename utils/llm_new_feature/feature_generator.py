"""
LLM特征生成主模块
负责批量生成特征并保存为JSON文件
"""
import json
import logging
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from .llm_api import call_llm_api_with_config, MAX_RETRIES, RETRY_DELAY
import json
from .llm_prompt import build_batch_feature_generation_prompt
from .llm_parser import parse_batch_llm_response, is_tree_duplicate
from .tree_height import calculate_tree_height

logger = logging.getLogger(__name__)


def generate_features(
    target_name: str,
    feature_names: List[str],
    num_features: int,
    task_context: str = None,
    output_dir: str = None,
    existing_trees: List[Dict] = None,
    api_config: Dict = None,
    return_raw_response: bool = False,
    max_height: int = None
) -> List[Dict]:
    """
    批量生成LLM特征（不自动保存）
    
    参数:
    target_name: 预测目标变量名
    feature_names: 特征名称列表
    num_features: 需要生成的特征数量
    task_context: 任务背景描述（完整的prompt）
    output_dir: 输出目录（未使用）
    existing_trees: 已存在的特征tree列表（用于避免重复）
    api_config: API配置字典，包含api_key, api_base_url, model, timeout, temperature, max_retries, retry_delay
    
    返回:
        生成的特征列表，每个元素包含tree, description, notation
    """
    # 获取已存在的特征tree列表（用于避免重复）
    if existing_trees is None:
        existing_trees = []
    
    # 使用默认API配置或自定义配置
    if api_config is None:
        api_config = {}
    
    generated_features = []
    generated_trees = []
    raw_responses = []  # 存储所有原始响应
    
    max_retries = api_config.get('max_retries', MAX_RETRIES)
    retry_delay = api_config.get('retry_delay', RETRY_DELAY)
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # 计算还需要生成的特征数量
            remaining_features = num_features - len(generated_features)
            if remaining_features <= 0:
                # 已经生成足够的特征，退出循环
                break
            
            # 构建批量生成提示词
            # 注意：使用remaining_features而不是num_features，告诉模型还需要生成多少个
            if task_context is None or task_context.strip() == "":
                # 如果没有提供prompt，使用默认构建方式
                prompt = build_batch_feature_generation_prompt(
                    target_name, feature_names, remaining_features, existing_trees + generated_trees, None, max_height
                )
            else:
                # 使用用户提供的完整prompt，但需要替换关键信息
                prompt = task_context
                import re
                # 替换特征数量（使用remaining_features而不是num_features）
                prompt = re.sub(r'(\d+)\s*个特征', f'{remaining_features} 个特征', prompt)
                prompt = re.sub(r'包含\s*(\d+)\s*个特征对象', f'包含 {remaining_features} 个特征对象', prompt)
                # 替换特征列表（如果prompt中有占位符）
                feature_list_str = ', '.join(feature_names)
                prompt = re.sub(r'可用特征列表[：:].*?(?=\n##|\n可用运算符|$)', 
                               f'可用特征列表（请严格使用这些名称，区分大小写）：\n{feature_list_str}', 
                               prompt, flags=re.DOTALL)
                # 替换目标变量
                prompt = re.sub(r'预测目标[：:]\s*\w+', f'预测目标：{target_name}', prompt)
            
            # 调用大模型API
            temperature = api_config.get('temperature', 0.8)
            # 从api_config中获取max_tokens，如果为None则不限制
            # 注意：None表示不限制，会传递None给API调用函数，由API调用函数决定如何处理
            max_tokens = api_config.get('max_tokens', None)
            response = call_llm_api_with_config(
                prompt, 
                max_tokens=max_tokens,  # 可以是None，表示不限制
                temperature=temperature,
                api_config=api_config
            )
            # 保存原始响应（无论成功或失败都要保存）
            if return_raw_response:
                if response is None:
                    # API调用失败，保存错误信息
                    raw_responses.append({
                        'attempt': attempt + 1,
                        'response': '[API调用失败：未收到响应]',
                        'timestamp': time.time(),
                        'error': True
                    })
                else:
                    # API调用成功，保存响应内容
                    raw_responses.append({
                        'attempt': attempt + 1,
                        'response': response,
                        'timestamp': time.time(),
                        'error': False
                    })
            
            if response is None:
                logger.warning(f"第 {attempt + 1} 次尝试：API调用失败")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
            
            logger.debug("=" * 80)
            logger.debug(f"【第 {attempt + 1} 次尝试】大模型返回的原始响应:")
            logger.debug("-" * 80)
            logger.debug(response[:500] + "..." if len(response) > 500 else response)
            logger.debug("-" * 80)
            
            # 解析批量响应
            parsed_results = parse_batch_llm_response(response)
            if parsed_results is None or len(parsed_results) == 0:
                logger.warning(f"第 {attempt + 1} 次尝试：批量响应解析失败或为空")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
            
            logger.debug(f"【第 {attempt + 1} 次尝试】解析到 {len(parsed_results)} 个特征")
            
            # 记录本次尝试成功添加的特征数量
            features_added_this_attempt = 0
            
            # 处理每个特征
            for i, parsed_result in enumerate(parsed_results):
                tree = parsed_result.get("tree")
                description = parsed_result.get("description", "")
                notation = parsed_result.get("notation", "")
                
                if tree is None:
                    logger.warning(f"第 {i+1} 个特征缺少tree字段，跳过")
                    continue
                
                # 计算表达式高度
                height = calculate_tree_height(tree)
                
                # 如果指定了高度限制，检查高度是否符合要求
                if max_height is not None and height != max_height:
                    logger.warning(f"第 {i+1} 个特征高度为 {height}，不符合要求的高度 {max_height}，跳过")
                    continue
                
                # 检查是否重复
                if is_tree_duplicate(tree, existing_trees + generated_trees):
                    logger.warning(f"第 {i+1} 个特征与已存在的特征重复，跳过")
                    continue
                
                # 添加特征（包含height字段）
                feature_data = {
                    "tree": tree,
                    "description": description,
                    "notation": notation,
                    "height": height
                }
                generated_features.append(feature_data)
                generated_trees.append(tree)
                features_added_this_attempt += 1
                
                logger.info(f"成功生成特征 {len(generated_features)}/{num_features}: {description[:50]}...")
                
                # 如果已经生成足够的特征，保存并返回
                if len(generated_features) >= num_features:
                    logger.info(f"已成功生成 {len(generated_features)} 个特征")
                    break
            
            # 如果这次尝试成功添加了一些特征（即使不够），也要保存并继续
            if features_added_this_attempt > 0:
                logger.info(f"第 {attempt + 1} 次尝试成功添加了 {features_added_this_attempt} 个特征，当前共有 {len(generated_features)}/{num_features} 个特征")
                # 检查是否已经生成足够的特征
                if len(generated_features) >= num_features:
                    logger.info(f"已成功生成 {len(generated_features)} 个特征")
                    break
                else:
                    # 还需要继续生成剩余的特征
                    remaining = num_features - len(generated_features)
                    logger.info(f"还需要生成 {remaining} 个特征，继续尝试...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
            else:
                # 这次尝试没有成功添加任何特征，继续重试
                logger.warning(f"第 {attempt + 1} 次尝试没有成功添加任何特征，继续重试")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
                
        except Exception as e:
            logger.error(f"第 {attempt + 1} 次尝试时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    
    # 如果需要返回原始响应，将其附加到返回值中
    if return_raw_response:
        return {
            'features': generated_features,
            'raw_responses': raw_responses
        }
    
    return generated_features


def save_features_to_json(
    features: List[Dict],
    target_name: str,
    feature_names: List[str],
    output_dir: str = None,
    model_name: str = None  # 已废弃：model_name现在包含在每个feature对象中
) -> Optional[str]:
    """
    将特征列表保存为JSON文件
    
    参数:
        features: 特征列表（每个特征对象应包含model_name字段）
        target_name: 目标变量名
        feature_names: 特征名称列表
        output_dir: 输出目录（如果为None，使用默认的json_save目录）
        model_name: 已废弃，不再使用（model_name现在包含在每个feature对象中）
    
    返回:
        保存的文件路径
    """
    if output_dir is None:
        # 默认保存到json_save目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'json_save')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_{target_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    output_data = {
        "target_name": target_name,
        "feature_names": feature_names,
        "num_features": len(features),
        "generated_at": timestamp,
        "features": features
    }
    
    # 注意：model_name已经包含在每个feature对象中，不需要在顶层添加
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 使用自定义JSON编码，确保feature_names和operands在一行显示
            json_str = json.dumps(output_data, ensure_ascii=False, indent=2)
            import re
            # 将feature_names数组格式化为一行
            pattern = r'"feature_names":\s*\[\s*(.*?)\s*\]'
            def replace_feature_names(match):
                content = match.group(1)
                # 移除所有换行、制表符和多余空格，但保留字符串之间的逗号和空格
                content = re.sub(r'\s+', ' ', content.strip())
                # 确保逗号后有空格
                content = re.sub(r',\s*', ', ', content)
                return f'"feature_names": [{content}]'
            json_str = re.sub(pattern, replace_feature_names, json_str, flags=re.DOTALL)
            
            # 将operands数组格式化为一行（只处理简单的字符串数组，不处理嵌套对象）
            # 匹配 "operands": [\n      "value1",\n      "value2"\n    ] 这种格式
            def format_simple_operands(match):
                full_match = match.group(0)
                # 检查是否包含嵌套对象（如果有{，说明是嵌套的，不处理）
                if '{' in full_match:
                    return full_match  # 嵌套对象保持原样
                # 提取数组内容
                content = match.group(1) if match.lastindex >= 1 else ''
                # 移除所有换行、制表符和多余空格
                content = re.sub(r'\s+', ' ', content.strip())
                # 确保逗号后有空格
                content = re.sub(r',\s*', ', ', content)
                return f'"operands": [{content}]'
            
            # 匹配operands数组（只匹配简单的字符串数组，不包含嵌套对象）
            operands_pattern = r'"operands":\s*\[\s*((?:"[^"]*"(?:\s*,\s*"[^"]*")*)?)\s*\]'
            json_str = re.sub(operands_pattern, format_simple_operands, json_str, flags=re.DOTALL)
            
            f.write(json_str)
        logger.info(f"特征已保存至: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"保存特征文件失败: {e}")
        return None

