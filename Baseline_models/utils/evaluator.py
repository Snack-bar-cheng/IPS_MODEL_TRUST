"""
模型训练和评估
"""

import os
import sys
import time
import warnings
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import export_text
from typing import List, Optional, Tuple

# 导入graphviz相关库（用于决策树可视化）
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    Digraph = None
    HAS_GRAPHVIZ = False
    try:
        import pygraphviz as pgv
        HAS_PYGRAPHVIZ = True
    except ImportError:
        pgv = None
        HAS_PYGRAPHVIZ = False

# 导入matplotlib相关库（用于线性模型系数可视化）
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

# 过滤 sklearn 的 RuntimeWarning 警告（溢出、无效值、除以零等）
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*overflow.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

# 导入SHAP工具（带回退）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UTILS_ROOT = os.path.join(PROJECT_ROOT, 'utils')
for p in (PROJECT_ROOT, UTILS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from utils.feature_importance.shap_handler import compute_shap_importances
    from utils.feature_importance.shap_sampler import sample_with_ks
except Exception as e:
    # 回退：直接按文件路径加载，避免包路径解析失败
    try:
        import importlib.util
        from pathlib import Path
        shap_handler_path = Path(UTILS_ROOT) / "feature_importance" / "shap_handler.py"
        shap_sampler_path = Path(UTILS_ROOT) / "feature_importance" / "shap_sampler.py"
        spec_h = importlib.util.spec_from_file_location("shap_handler_local", shap_handler_path)
        spec_s = importlib.util.spec_from_file_location("shap_sampler_local", shap_sampler_path)
        shap_handler = importlib.util.module_from_spec(spec_h)
        shap_sampler = importlib.util.module_from_spec(spec_s)
        spec_h.loader.exec_module(shap_handler)  # type: ignore
        spec_s.loader.exec_module(shap_sampler)  # type: ignore
        compute_shap_importances = shap_handler.compute_shap_importances
        sample_with_ks = shap_sampler.sample_with_ks
        print("[SHAP][INFO] baseline模块采用文件路径回退方式导入SHAP工具")
    except Exception as e2:
        print(f"[SHAP][WARNING] baseline模块导入SHAP工具失败: {e}; fallback失败: {e2}")
        compute_shap_importances = None
        sample_with_ks = None


def calculate_metrics(y_true, y_pred):
    """计算四个评估指标"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, mae


def visualize_decision_tree(model, model_name: str, feature_names: List[str], target_name: str, save_dir: str, random_seed: int = 1):
    """
    可视化决策树模型并保存为PDF
    
    参数:
        model: 训练好的决策树模型（DecisionTreeRegressor）
        model_name: 模型名称（支持'DecisionTree'和'DecisionTree_4'）
        feature_names: 特征名称列表
        target_name: 目标变量名称
        save_dir: 保存目录路径
        random_seed: 随机种子（用于文件名）
    
    返回:
        str: PDF文件路径，如果失败则返回None
    """
    if model_name not in ['DecisionTree', 'DecisionTree_2', 'DecisionTree_4', 'DecisionTree_6']:
        return None
    
    if not HAS_GRAPHVIZ and not HAS_PYGRAPHVIZ:
        print("[WARNING] graphviz or pygraphviz not installed, cannot generate decision tree visualization")
        return None
    
    try:
        # 检查模型是否有tree_属性
        if not hasattr(model, 'tree_'):
            return None
        
        # 创建baseline_model目录
        # save_dir应该是target_dir，即类似 /path/to/gps_TGWO_种群100_50代_8/Ash_Deformation/Ash_Deformation
        # 需要在save_dir下创建baseline_model文件夹
        baseline_model_dir = os.path.join(save_dir, 'baseline_model')
        os.makedirs(baseline_model_dir, exist_ok=True)
        
        # 如果没有提供特征名称，使用默认名称
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        # 确保特征名称数量匹配
        if len(feature_names) != model.n_features_in_:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        tree = model.tree_
        
        # 定义颜色方案（与GP树保持一致）
        DECISION_NODE_COLOR = '#FFCC99'  # 决策节点（浅橙色，类似GP函数节点）
        LEAF_NODE_COLOR = '#CDEB8B'      # 叶节点（浅绿色，类似GP终端节点）
        
        if HAS_GRAPHVIZ:
            # 使用graphviz库
            dot = Digraph(format='pdf')
            dot.attr('node', shape='box', style='filled')
            dot.attr('graph', rankdir='TB')  # 从上到下布局
            
            def add_node(node_id, parent_id=None, edge_label=''):
                """递归添加节点"""
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # 叶节点
                    # tree.value[node_id] 是一个数组，需要提取标量值
                    value_array = tree.value[node_id]
                    if isinstance(value_array, np.ndarray):
                        # 对于回归问题，取第一个元素
                        value = float(value_array[0]) if value_array.size > 0 else 0.0
                    else:
                        value = float(value_array)
                    samples = int(tree.n_node_samples[node_id])
                    label = f"Value: {value:.4f}\\nSamples: {samples}"
                    dot.node(str(node_id), label, fillcolor=LEAF_NODE_COLOR)
                else:
                    # 决策节点
                    feature_idx = int(tree.feature[node_id])
                    threshold = float(tree.threshold[node_id])
                    samples = int(tree.n_node_samples[node_id])
                    label = f"{feature_names[feature_idx]}\\n<= {threshold:.4f}\\nSamples: {samples}"
                    dot.node(str(node_id), label, fillcolor=DECISION_NODE_COLOR)
                    
                    # 递归添加子节点
                    add_node(tree.children_left[node_id], node_id, 'True')
                    add_node(tree.children_right[node_id], node_id, 'False')
                    
                    # 添加边
                    dot.edge(str(node_id), str(tree.children_left[node_id]), label='True')
                    dot.edge(str(node_id), str(tree.children_right[node_id]), label='False')
                
                # 如果有父节点，添加边
                if parent_id is not None and edge_label:
                    pass  # 边已经在递归中添加
            
            add_node(0)
            
            # 生成文件名（使用模型名称）
            filename = f"{model_name}_{target_name}_{random_seed}"
            filepath = os.path.join(baseline_model_dir, filename)
            dot.render(filepath, cleanup=True)  # 生成 filepath.pdf
            pdf_path = f"{filepath}.pdf"
            
            return pdf_path
            
        elif HAS_PYGRAPHVIZ:
            # 使用pygraphviz库（备用方案）
            graph = pgv.AGraph(directed=True)
            graph.attr(rankdir='TB')
            
            def add_node_pgv(node_id):
                """递归添加节点（pygraphviz版本）"""
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # 叶节点
                    # tree.value[node_id] 是一个数组，需要提取标量值
                    value_array = tree.value[node_id]
                    if isinstance(value_array, np.ndarray):
                        # 对于回归问题，取第一个元素
                        value = float(value_array[0]) if value_array.size > 0 else 0.0
                    else:
                        value = float(value_array)
                    samples = int(tree.n_node_samples[node_id])
                    label = f"Value: {value:.4f}\\nSamples: {samples}"
                    graph.add_node(node_id, label=label, style='filled', fillcolor=LEAF_NODE_COLOR)
                else:
                    # 决策节点
                    feature_idx = int(tree.feature[node_id])
                    threshold = float(tree.threshold[node_id])
                    samples = int(tree.n_node_samples[node_id])
                    label = f"{feature_names[feature_idx]}\\n<= {threshold:.4f}\\nSamples: {samples}"
                    graph.add_node(node_id, label=label, style='filled', fillcolor=DECISION_NODE_COLOR)
                    
                    # 递归添加子节点
                    add_node_pgv(tree.children_left[node_id])
                    add_node_pgv(tree.children_right[node_id])
                    
                    # 添加边
                    graph.add_edge(node_id, tree.children_left[node_id], label='True')
                    graph.add_edge(node_id, tree.children_right[node_id], label='False')
            
            add_node_pgv(0)
            graph.layout(prog='dot')
            
            # 生成文件名（使用模型名称）
            filename = f"{model_name}_{target_name}_{random_seed}.pdf"
            filepath = os.path.join(baseline_model_dir, filename)
            graph.draw(filepath)
            
            return filepath
        
    except Exception as e:
        print(f"[WARNING] Failed to generate {model_name} visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_decision_tree_rules(model, model_name: str, feature_names: List[str], target_name: str, max_depth: Optional[int] = None):
    """
    为决策树模型生成规则表达式
    
    参数:
        model: 训练好的决策树模型（DecisionTreeRegressor）
        model_name: 模型名称（支持'DecisionTree', 'DecisionTree_2', 'DecisionTree_4', 'DecisionTree_6'）
        feature_names: 特征名称列表
        target_name: 目标变量名称
        max_depth: 最大深度（可选），如果为None则使用模型的实际深度
    
    返回:
        str: 规则表达式字符串，格式为决策树的文本表示
        如果模型不支持或出错，返回None
    """
    if model_name not in ['DecisionTree', 'DecisionTree_2', 'DecisionTree_4', 'DecisionTree_6']:
        return None
    
    try:
        # 检查模型是否有tree_属性（sklearn决策树的标准属性）
        if not hasattr(model, 'tree_'):
            return None
        
        # 如果没有提供特征名称，使用默认名称
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        # 确保特征名称数量匹配
        if len(feature_names) != model.n_features_in_:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        # 使用sklearn的export_text函数导出决策树规则
        # max_depth参数控制显示的深度，如果为None则显示全部
        tree_rules = export_text(
            model,
            feature_names=feature_names,
            max_depth=max_depth,
            spacing=2,
            decimals=6,
            show_weights=True
        )
        
        # 添加目标变量名称作为前缀，使规则更清晰
        rules_with_target = f"{target_name} Decision Rules:\n{tree_rules}"
        
        return rules_with_target
        
    except Exception as e:
        # 如果生成规则失败，返回None
        print(f"[WARNING] Failed to generate {model_name} rules: {e}")
        return None


def visualize_linear_model_tree(model, model_name: str, feature_names: List[str], target_name: str, save_dir: str, random_seed: int = 1):
    """
    可视化线性模型为树状图（类似GP树）并保存为PDF
    
    参数:
        model: 训练好的线性模型（LinearRegression、RidgeCV或ElasticNet）
        model_name: 模型名称
        feature_names: 特征名称列表
        target_name: 目标变量名称
        save_dir: 保存目录路径
        random_seed: 随机种子（用于文件名）
    
    返回:
        str: PDF文件路径，如果失败则返回None
    """
    if model_name not in ['LinearRegression', 'RidgeCV', 'ElasticNet']:
        return None
    
    if not HAS_GRAPHVIZ and not HAS_PYGRAPHVIZ:
        print("[WARNING] graphviz or pygraphviz not installed, cannot generate linear model tree visualization")
        return None
    
    try:
        # 检查模型是否有coef_和intercept_属性
        if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
            return None
        
        # 创建baseline_model目录
        baseline_model_dir = os.path.join(save_dir, 'baseline_model')
        os.makedirs(baseline_model_dir, exist_ok=True)
        
        # 获取系数和截距
        coefficients = model.coef_
        intercept = float(model.intercept_)
        
        # 如果coef_是1维数组，需要处理
        if coefficients.ndim == 1:
            coefficients = coefficients
        else:
            # 如果是2维（多输出），取第一列
            coefficients = coefficients[:, 0] if coefficients.shape[1] == 1 else coefficients[0]
        
        # 确保特征名称数量与系数数量匹配
        if len(feature_names) != len(coefficients):
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        # 提取非零系数项
        terms = []
        for coef, feature_name in zip(coefficients, feature_names):
            coef_val = float(coef)
            if abs(coef_val) > 1e-10:  # 只显示非零系数
                terms.append((coef_val, feature_name))
        
        # 定义颜色方案（与GP树保持一致）
        FUNCTION_COLOR = '#FFCC99'  # 函数节点（浅橙色，类似GP函数节点）
        TERMINAL_COLOR = '#CDEB8B'  # 终端节点（浅绿色，类似GP终端节点）
        
        # 处理化学式下标：将数字转换为下标格式（使用HTML标签）
        # 例如：SiO2 -> SiO<sub>2</sub>, Al2O3 -> Al<sub>2</sub>O<sub>3</sub>
        import re
        def format_chemical_formula(text):
            """将化学式中的数字转换为下标"""
            if not text:
                return text
            # 匹配化学式中的数字（前面是字母，后面可能是字母或结束）
            # 例如：SiO2, Al2O3, Fe2O3等
            # 使用正则表达式匹配：字母后跟数字，数字后可能是字母或结束
            pattern = r'([A-Za-z]+)(\d+)(?=[A-Za-z]|$)'
            def replace_chem_subscript(match):
                letters = match.group(1)
                nums = match.group(2)
                # 将数字转换为下标
                subscript_nums = ''.join([f'<sub>{n}</sub>' for n in nums])
                return letters + subscript_nums
            result = re.sub(pattern, replace_chem_subscript, text)
            # 处理末尾的数字（如果存在）
            if result != text:
                return result
            # 如果没有匹配到，尝试匹配末尾的数字
            pattern_end = r'([A-Za-z]+)(\d+)$'
            result = re.sub(pattern_end, replace_chem_subscript, text)
            return result
        
        # 构建完整公式字符串（用于High节点显示）
        formula_parts = []
        # 先添加特征项
        for coef_val, feature_name in terms:
            if coef_val >= 0:
                sign = " + "
            else:
                sign = " - "
                coef_val = abs(coef_val)
            # 使用乘号（×）而不是星号（*），并处理化学式下标
            formatted_feature = format_chemical_formula(feature_name)
            formula_parts.append(f"{sign}{coef_val:.4f} × {formatted_feature}")
        
        # 最后添加截距（如果有）
        if abs(intercept) > 1e-10:
            if intercept >= 0:
                sign = " + "
                intercept_val = intercept
            else:
                sign = " - "
                intercept_val = abs(intercept)
            formula_parts.append(f"{sign}{intercept_val:.4f}")
        
        if formula_parts:
            # 处理目标变量名称中的化学式
            formatted_target = format_chemical_formula(target_name)
            formula_str = f"{formatted_target} = " + "".join(formula_parts)
        else:
            formatted_target = format_chemical_formula(target_name)
            formula_str = f"{formatted_target} = 0.0"
        
        # 构建所有输入项（特征和截距都视为输入）
        all_inputs = []
        # 先添加特征项
        for coef_val, feature_name in terms:
            all_inputs.append((coef_val, feature_name, False))  # (value, name, is_intercept)
        # 再添加截距项（如果有）
        if abs(intercept) > 1e-10:
            all_inputs.append((intercept, None, True))  # (value, None, is_intercept)
        
        # 计算High函数的n值（输入数量）
        high_n = len(all_inputs)
        
        if HAS_GRAPHVIZ:
            dot = Digraph(format='pdf')
            dot.attr('node', shape='ellipse', style='filled')  # 使用椭圆形状，与GP树一致
            dot.attr('graph', rankdir='TB')  # 从上到下布局
            
            # 根节点：High_n函数（不包含公式）
            root_id = "root"
            high_label = f"High_{high_n}"
            dot.node(root_id, high_label, fillcolor=FUNCTION_COLOR)
            
            # 添加所有输入节点（特征和截距）
            for idx, (val, feat_name, is_intercept) in enumerate(all_inputs):
                if is_intercept:
                    # 截距节点
                    input_id = f"input_{idx}"
                    dot.node(input_id, f"{val:.4f}", fillcolor=TERMINAL_COLOR)
                else:
                    # 特征节点（应用化学式格式化）
                    input_id = f"input_{idx}"
                    formatted_feat_name = format_chemical_formula(feat_name)
                    # 如果包含HTML标签，需要使用HTML格式
                    if '<sub>' in formatted_feat_name:
                        dot.node(input_id, f'<{formatted_feat_name}>', fillcolor=TERMINAL_COLOR)
                    else:
                        dot.node(input_id, formatted_feat_name, fillcolor=TERMINAL_COLOR)
                dot.edge(root_id, input_id)
            
            # 公式节点（独立显示在最下面，使用方框和粉色背景）
            # 使用HTML标签来支持下标显示
            formula_id = "formula"
            # Graphviz支持HTML标签：如果formula_str包含HTML标签（如<sub>），需要包装在<...>中
            # 如果formula_str已经包含HTML标签，直接使用；否则需要包装
            if '<sub>' in formula_str or '<SUP>' in formula_str.upper():
                # 已经包含HTML标签，直接使用
                html_formula = f'<{formula_str}>'
            else:
                # 没有HTML标签，直接使用原字符串
                html_formula = formula_str
            dot.node(formula_id, html_formula, shape='box', style='filled', fillcolor='#FFB6C1')  # 粉色背景
            
            # 使用rank='sink'来确保公式节点在最下方
            with dot.subgraph() as sink_subgraph:
                sink_subgraph.attr(rank='sink')
                sink_subgraph.node(formula_id)
            
            # 使用不可见边确保公式节点在所有输入节点下方，并水平对齐
            # 连接到第一个和最后一个输入节点，确保公式在下方并水平对齐
            if all_inputs:
                first_input_id = f"input_0"
                last_input_id = f"input_{len(all_inputs) - 1}"
                # 连接到第一个和最后一个节点，确保水平对齐
                dot.edge(first_input_id, formula_id, style='invis', weight='0', constraint='false')
                dot.edge(last_input_id, formula_id, style='invis', weight='0', constraint='false')
            
            # 生成文件名
            filename = f"{model_name}_tree_{target_name}_{random_seed}"
            filepath = os.path.join(baseline_model_dir, filename)
            dot.render(filepath, cleanup=True)  # 生成 filepath.pdf
            pdf_path = f"{filepath}.pdf"
            
            return pdf_path
            
        elif HAS_PYGRAPHVIZ:
            # pygraphviz版本（类似实现）
            graph = pgv.AGraph(directed=True)
            graph.attr(rankdir='TB')
            
            # 根节点：High_n函数（不包含公式）
            root_id = "root"
            high_label = f"High_{high_n}"
            graph.add_node(root_id, label=high_label, shape='ellipse', style='filled', fillcolor=FUNCTION_COLOR)
            
            # 添加所有输入节点（特征和截距）
            for idx, (val, feat_name, is_intercept) in enumerate(all_inputs):
                if is_intercept:
                    # 截距节点
                    input_id = f"input_{idx}"
                    graph.add_node(input_id, label=f"{val:.4f}", shape='ellipse', style='filled', fillcolor=TERMINAL_COLOR)
                else:
                    # 特征节点（应用化学式格式化）
                    input_id = f"input_{idx}"
                    formatted_feat_name = format_chemical_formula(feat_name)
                    # 如果包含HTML标签，需要使用HTML格式
                    if '<sub>' in formatted_feat_name:
                        graph.add_node(input_id, label=f'<{formatted_feat_name}>', shape='ellipse', style='filled', fillcolor=TERMINAL_COLOR)
                    else:
                        graph.add_node(input_id, label=formatted_feat_name, shape='ellipse', style='filled', fillcolor=TERMINAL_COLOR)
                graph.add_edge(root_id, input_id)
            
            # 公式节点（独立显示在最下面，使用方框和粉色背景）
            # 使用HTML标签来支持下标显示
            formula_id = "formula"
            # pygraphviz也支持HTML标签：如果formula_str包含HTML标签，需要包装在<...>中
            if '<sub>' in formula_str or '<SUP>' in formula_str.upper():
                # 已经包含HTML标签，直接使用
                html_formula = f'<{formula_str}>'
            else:
                # 没有HTML标签，直接使用原字符串
                html_formula = formula_str
            graph.add_node(formula_id, label=html_formula, shape='box', style='filled', fillcolor='#FFB6C1')  # 粉色背景
            
            # 使用不可见边来确保公式节点在最下方，并水平对齐
            if all_inputs:
                first_input_id = f"input_0"
                last_input_id = f"input_{len(all_inputs) - 1}"
                # 连接到第一个和最后一个节点，确保水平对齐
                graph.add_edge(first_input_id, formula_id, style='invis')
                graph.add_edge(last_input_id, formula_id, style='invis')
            
            # 生成文件名
            filename = f"{model_name}_tree_{target_name}_{random_seed}"
            filepath = os.path.join(baseline_model_dir, filename)
            graph.draw(f"{filepath}.pdf", prog='dot')
            pdf_path = f"{filepath}.pdf"
            
            return pdf_path
        
    except Exception as e:
        print(f"[WARNING] Failed to generate {model_name} tree visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_linear_model_coefficients(model, model_name: str, feature_names: List[str], target_name: str, save_dir: str, random_seed: int = 1):
    """
    可视化线性模型的系数并保存为PDF
    
    参数:
        model: 训练好的线性模型（LinearRegression、RidgeCV或ElasticNet）
        model_name: 模型名称
        feature_names: 特征名称列表
        target_name: 目标变量名称
        save_dir: 保存目录路径
        random_seed: 随机种子（用于文件名）
    
    返回:
        str: PDF文件路径，如果失败则返回None
    """
    if model_name not in ['LinearRegression', 'RidgeCV', 'ElasticNet']:
        return None
    
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not installed, cannot generate linear model coefficient visualization")
        return None
    
    try:
        # 检查模型是否有coef_和intercept_属性
        if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
            return None
        
        # 创建baseline_model目录
        baseline_model_dir = os.path.join(save_dir, 'baseline_model')
        os.makedirs(baseline_model_dir, exist_ok=True)
        
        # 获取系数和截距
        coefficients = model.coef_
        intercept = float(model.intercept_)
        
        # 如果coef_是1维数组，需要处理
        if coefficients.ndim == 1:
            coefficients = coefficients
        else:
            # 如果是2维（多输出），取第一列
            coefficients = coefficients[:, 0] if coefficients.shape[1] == 1 else coefficients[0]
        
        # 确保特征名称数量与系数数量匹配
        if len(feature_names) != len(coefficients):
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        # 转换为numpy数组并提取非零系数
        coef_array = np.array([float(c) for c in coefficients])
        abs_coef = np.abs(coef_array)
        
        # 按系数绝对值排序（降序）
        sorted_indices = np.argsort(abs_coef)[::-1]
        sorted_coef = coef_array[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_abs_coef = abs_coef[sorted_indices]
        
        # 只显示前20个最重要的系数（如果特征太多）
        max_features = 20
        if len(sorted_coef) > max_features:
            sorted_coef = sorted_coef[:max_features]
            sorted_features = sorted_features[:max_features]
            sorted_abs_coef = sorted_abs_coef[:max_features]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_coef) * 0.4)))
        
        # 设置颜色：正系数为蓝色，负系数为红色
        colors = ['#4169E1' if c >= 0 else '#DC143C' for c in sorted_coef]
        
        # 创建水平条形图
        y_pos = np.arange(len(sorted_coef))
        bars = ax.barh(y_pos, sorted_coef, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 设置y轴标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features, fontsize=9)
        ax.invert_yaxis()  # 最重要的特征在顶部
        
        # 设置x轴标签
        ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} - {target_name} Coefficient Visualization\nIntercept: {intercept:.6f}', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # 添加网格线
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        # 在条形图上添加数值标签
        for i, (bar, coef_val) in enumerate(zip(bars, sorted_coef)):
            width = bar.get_width()
            label_x = width + (0.02 * max(abs_coef)) if width >= 0 else width - (0.02 * max(abs_coef))
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{coef_val:.4f}', 
                   ha='left' if width >= 0 else 'right',
                   va='center', fontsize=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存PDF
        filename = f"{model_name}_{target_name}_{random_seed}.pdf"
        filepath = os.path.join(baseline_model_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"[WARNING] Failed to generate {model_name} coefficient visualization: {e}")
        import traceback
        traceback.print_exc()
        if HAS_MATPLOTLIB:
            plt.close('all')  # 确保关闭所有图表
        return None


def generate_linear_model_formula(model, model_name: str, feature_names: List[str], target_name: str):
    """
    为线性模型生成公式字符串
    
    参数:
        model: 训练好的线性模型（LinearRegression、RidgeCV或ElasticNet）
        model_name: 模型名称
        feature_names: 特征名称列表
        target_name: 目标变量名称
    
    返回:
        str: 公式字符串，格式如 "target_name = intercept + coef1 * feature1 + coef2 * feature2 ..."
        如果模型不支持或出错，返回None
    """
    # 只支持这三种线性模型
    supported_models = ['LinearRegression', 'RidgeCV', 'ElasticNet']
    if model_name not in supported_models:
        return None
    
    try:
        # 检查模型是否有coef_和intercept_属性
        if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
            return None
        
        # 获取系数和截距
        coefficients = model.coef_
        intercept = float(model.intercept_)  # 确保是标量
        
        # 如果coef_是1维数组，需要处理
        if coefficients.ndim == 1:
            coefficients = coefficients
        else:
            # 如果是2维（多输出），取第一列
            coefficients = coefficients[:, 0] if coefficients.shape[1] == 1 else coefficients[0]
        
        # 确保特征名称数量与系数数量匹配
        if len(feature_names) != len(coefficients):
            # 如果特征名称数量不匹配，使用默认名称
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        # 构建公式字符串
        formula_parts = []
        
        # 添加截距项（如果不为0）
        if abs(intercept) > 1e-10:
            formula_parts.append(f"{intercept:.6f}")
        
        # 添加特征项
        for coef, feature_name in zip(coefficients, feature_names):
            coef_val = float(coef)  # 确保是标量
            if abs(coef_val) > 1e-10:  # 只显示非零系数
                if coef_val >= 0:
                    sign = " + "
                else:
                    sign = " - "
                    coef_val = abs(coef_val)
                formula_parts.append(f"{sign}{coef_val:.6f} * {feature_name}")
        
        # 组合公式
        if formula_parts:
            formula = f"{target_name} = " + "".join(formula_parts)
        else:
            formula = f"{target_name} = 0.0"
        
        return formula
        
    except Exception as e:
        # 如果生成公式失败，返回None
        print(f"[WARNING] Failed to generate {model_name} formula: {e}")
        return None


def extract_feature_importances(model, model_name: str, feature_names: Optional[List[str]] = None):
    """
    提取模型的特征重要性（使用传统的 feature_importances_）
    
    参数:
        model: 训练好的模型实例
        model_name: 模型名称
        feature_names: 特征名称列表
    
    返回:
        (feature_importances, interaction_importances): 
        - feature_importances: 特征重要性列表或 None
        - interaction_importances: 交互特征重要性列表或 None（始终为 None）
    """
    try:
        # 检查模型是否有 feature_importances_ 属性
        if not hasattr(model, 'feature_importances_'):
            return None, None
        
        # 获取特征重要性数组
        importances = model.feature_importances_
        
        # 如果没有提供特征名称，使用索引
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # 确保特征名称和重要性数量匹配
        if len(feature_names) != len(importances):
            return None, None
        
        # 创建特征重要性列表，按重要性降序排列
        feature_importance_list = [
            {"importance": float(imp), "feature_name": name}
            for imp, name in zip(importances, feature_names)
        ]
        
        # 按重要性降序排序
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)
        
        return feature_importance_list, None
    except Exception:
        # 如果提取失败，返回 None
        return None, None


def train_and_evaluate_model(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    cv_folds: int = 5,
    random_seed: int = 1,
    feature_names: Optional[List[str]] = None,
    shap_open: bool = False,
    shap_save_dir: Optional[str] = None,
    target_name: str = "",
    shap_bg_size: Optional[int] = None,
    shap_explain_size: Optional[int] = None,
    shap_ks_threshold: float = 0.05,
    shap_max_attempts: int = 5,
    target_dir: Optional[str] = None,  # 新增：目标目录路径，用于保存可视化文件
):
    """
    训练模型并在训练集上进行K折交叉验证，另外评估测试集性能
    
    参数:
        model: 模型实例
        model_name: 模型名称
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        cv_folds: 交叉验证折数
        random_seed: 随机种子
        feature_names: 特征名称列表（用于特征重要性提取）
    
    返回:
        dict: 包含模型评估结果的字典
    """
    # 训练集K折交叉验证（仅在训练集上）
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_metrics = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model_fold = clone(model)
        model_fold.fit(X_tr, y_tr)
        y_val_pred = model_fold.predict(X_val)
        r2, mse, rmse, mae = calculate_metrics(y_val, y_val_pred)
        fold_metrics.append({'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae})

    # 计算CV均值与标准差（适配baseline_executor期望的格式）
    cv_summary = {
        'r2_mean': float(np.mean([m['r2'] for m in fold_metrics])),
        'r2_std': float(np.std([m['r2'] for m in fold_metrics])),
        'mse_mean': float(np.mean([m['mse'] for m in fold_metrics])),
        'mse_std': float(np.std([m['mse'] for m in fold_metrics])),
        'rmse_mean': float(np.mean([m['rmse'] for m in fold_metrics])),
        'rmse_std': float(np.std([m['rmse'] for m in fold_metrics])),
        'mae_mean': float(np.mean([m['mae'] for m in fold_metrics])),
        'mae_std': float(np.std([m['mae'] for m in fold_metrics]))
    }

    # 使用全部训练集拟合并评估训练/测试集
    # 记录最终模型训练时间
    final_train_start_time = time.time()
    model.fit(X_train, y_train)
    final_train_end_time = time.time()
    final_training_time = final_train_end_time - final_train_start_time
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2, train_mse, train_rmse, train_mae = calculate_metrics(y_train, y_train_pred)
    test_r2, test_mse, test_rmse, test_mae = calculate_metrics(y_test, y_test_pred)

    # 计算训练时长的不同单位
    training_seconds = float(final_training_time)
    training_minutes = training_seconds / 60.0
    training_hours = training_seconds / 3600.0

    # 提取特征重要性
    feature_importances = None
    interaction_importances = None
    shap_plot_path = None
    shap_bee_path = None

    # 调试信息
    if shap_open:
        print(f"[SHAP][DEBUG] baseline阶段: shap_open={shap_open}, compute_shap_importances={compute_shap_importances is not None}, model_name={model_name}")

    if shap_open and compute_shap_importances is not None:
        print(f"[SHAP] 正在计算baseline模型的SHAP值... 模型={model_name}, 目标={target_name}")

        X_bg = X_train
        X_explain = X_test
        if sample_with_ks is not None:
            X_bg, X_explain = sample_with_ks(
                X_train, X_test,
                background_size=shap_bg_size,
                explain_size=shap_explain_size,
                ks_threshold=shap_ks_threshold,
                max_attempts=shap_max_attempts,
                random_state=random_seed
            )
        
        # 输出背景集和解释集的数据条数
        print(f"[SHAP] 背景集数据条数: {X_bg.shape[0]} (配置: {shap_bg_size if shap_bg_size is not None else 'None-使用全部训练集'})")
        print(f"[SHAP] 解释集数据条数: {X_explain.shape[0]} (配置: {shap_explain_size if shap_explain_size is not None else 'None-使用全部测试集'})")

        fi_payload, inter_payload, shap_plot_path, shap_bee_path = compute_shap_importances(
            model=model,
            X_train=X_bg,
            X_explain=X_explain,
            feature_names=feature_names,
            target_name=target_name,
            model_name=model_name,
            random_seed=random_seed,
            save_dir=shap_save_dir,
            prefix="baseline",
            background_limit=None,  # None表示使用全部采样后的背景数据
            explain_limit=None,     # None表示使用全部采样后的解释数据
        )
        if fi_payload is not None:
            feature_importances = fi_payload
        if inter_payload is not None:
            interaction_importances = inter_payload

    # 如果SHAP未开启或失败，回退到其他方法
    if feature_importances is None:
        # 优先使用模型自带的方法
        if hasattr(model, "compute_feature_importances"):
            try:
                feature_importances = model.compute_feature_importances(
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                )
            except Exception as e:  # pragma: no cover - 容错处理
                print(f"  Feature importance calculation failed (model built-in method): {e}")
                feature_importances = None
                interaction_importances = None

        # 回退到通用的 feature_importances_ 逻辑（树模型等）
        if feature_importances is None:
            feature_importances, interaction_importances = extract_feature_importances(
                model=model,
                model_name=model_name,
                feature_names=feature_names
            )

    # 生成显示表达（公式或规则或可视化路径，根据模型类型）
    display_expression = None
    if model_name in ['LinearRegression', 'RidgeCV', 'ElasticNet']:
        # 线性模型：优先生成树状图，然后生成条形图，最后回退到文本公式
        if target_dir is not None:
            # 优先尝试生成树状图（类似GP树）
            tree_pdf_path = visualize_linear_model_tree(
                model=model,
                model_name=model_name,
                feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                target_name=target_name,
                save_dir=target_dir,
                random_seed=random_seed
            )
            if tree_pdf_path:
                display_expression = tree_pdf_path
            else:
                # 如果树状图失败，尝试生成条形图
                bar_pdf_path = visualize_linear_model_coefficients(
                    model=model,
                    model_name=model_name,
                    feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                    target_name=target_name,
                    save_dir=target_dir,
                    random_seed=random_seed
                )
                if bar_pdf_path:
                    display_expression = bar_pdf_path
                else:
                    # 如果可视化都失败，回退到文本公式
                    display_expression = generate_linear_model_formula(
                        model=model,
                        model_name=model_name,
                        feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                        target_name=target_name
                    )
        else:
            # 如果没有target_dir，只生成文本公式
            display_expression = generate_linear_model_formula(
                model=model,
                model_name=model_name,
                feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                target_name=target_name
            )
    elif model_name in ['DecisionTree', 'DecisionTree_2', 'DecisionTree_4', 'DecisionTree_6']:
        # 决策树：生成可视化PDF并保存路径
        if target_dir is not None:
            # 生成可视化PDF
            pdf_path = visualize_decision_tree(
                model=model,
                model_name=model_name,
                feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                target_name=target_name,
                save_dir=target_dir,
                random_seed=random_seed
            )
            if pdf_path:
                display_expression = pdf_path
            else:
                # 如果可视化失败，回退到文本规则
                display_expression = generate_decision_tree_rules(
                    model=model,
                    model_name=model_name,
                    feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                    target_name=target_name,
                    max_depth=None
                )
        else:
            # 如果没有target_dir，只生成文本规则
            display_expression = generate_decision_tree_rules(
                model=model,
                model_name=model_name,
                feature_names=feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_train.shape[1])],
                target_name=target_name,
                max_depth=None
            )

    # 返回适配baseline_executor期望的格式
    result = {
        'model_name': model_name,
        'display_expression': display_expression,  # 显示表达：公式、规则或可视化文件路径（放在model_name之后）
        'cv_metrics': {
            'folds': cv_folds,
            'cross_validation': cv_summary
        },
        'train_metrics': {
            'r2': float(train_r2),
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'mae': float(train_mae)
        },
        'test_metrics': {
            'r2': float(test_r2),
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'mae': float(test_mae)
        },
        'training_duration': {
            'seconds': training_seconds,
            'minutes': training_minutes,
            'hours': training_hours
        },
        'feature_importances': feature_importances if feature_importances is not None else None,
        'interaction_feature_importances': interaction_importances if interaction_importances is not None else None
    }
    if shap_plot_path:
        result['shap_plot_path'] = shap_plot_path
    if shap_bee_path:
        result['shap_beeswarm_plot_path'] = shap_bee_path
    
    return result

