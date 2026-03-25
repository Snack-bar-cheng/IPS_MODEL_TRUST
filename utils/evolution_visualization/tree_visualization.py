"""
GP树可视化模块
包含用于绘制GP树的可视化功能
"""

try:
    import pygraphviz as pgv
    HAS_PYGRAPHVIZ = True
except Exception:
    pgv = None
    HAS_PYGRAPHVIZ = False
try:
    from graphviz import Digraph as _GraphvizDigraph  # type: ignore
    HAS_GRAPHVIZ = True
except Exception:
    _GraphvizDigraph = None
    HAS_GRAPHVIZ = False
from collections import Counter
from deap import gp
import os
import logging
import re
from .tree_matching import _find_matching_llm_features


def format_chemical_formula(text):
    """将化学式中的数字转换为下标格式（使用HTML标签）"""
    if not text:
        return text
    # 匹配化学式中的数字（前面是字母，后面可能是字母或结束）
    # 例如：SiO2, Al2O3, Fe2O3等
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


def Plot_tree(tree, randomSeeds, tree_save_fileplace, feature_names, generation_i, target_name=None, llm_features=None, is_last_generation=False, ridge_formula=None):
    """
    绘制GP树的可视化图形
    
    参数:
        tree: GP树个体
        randomSeeds: 随机种子
        tree_save_fileplace: 树保存路径
        feature_names: 特征名称列表
        generation_i: 代数索引
        target_name: 目标变量名称（可选）
        llm_features: LLM特征列表（可选），用于在最后一代可视化时比对
        is_last_generation: 是否为最后一代（默认False），如果为True且提供了llm_features，将比对特征并标记匹配的节点
        ridge_formula: Ridge回归公式（可选），如果提供，将在PDF上显示
    """
    feature_count_list = []

    generation_i += 1
    # 生成绘制GP树所需的元素
    nodes, edges, labels = gp.graph(tree)

    # 打印最好的个体所使用的特征和数字
    value_counts = Counter(labels.values())
    count_feature = [value_counts.get(name, 0) for name in feature_names]

    feature_count_list.append(count_feature)

    for node, label in labels.items():
        if isinstance(tree[node], gp.Terminal):
            if isinstance(tree[node].value, float):
                tree[node].value = round(tree[node].value, 4)
                labels[node] = tree[node].value

    # 定义颜色方案（统一使用原有颜色）
    TERMINAL_COLOR = '#CDEB8B'  # 终端节点（浅绿色）
    FUNCTION_COLOR = '#FFCC99'  # 函数节点（浅橙色）
    
    # LLM特征匹配颜色（用于标记与LLM特征匹配的节点）
    LLM_TERMINAL_COLOR = '#B0E0E6'  # 终端节点（浅蓝色）
    LLM_FUNCTION_COLOR = '#DDA0DD'  # 函数节点（浅紫色）

    # 确保保存目录存在
    os.makedirs(tree_save_fileplace, exist_ok=True)
    
    # 如果是最后一代且有LLM特征，查找匹配的节点
    matching_nodes = set()
    logger = logging.getLogger(__name__)
    if is_last_generation and llm_features:
        logger.info(f"开始匹配LLM特征，LLM特征数量: {len(llm_features)}")
        matching_nodes = _find_matching_llm_features(tree, llm_features, feature_names)
        logger.info(f"匹配完成，找到 {len(matching_nodes)} 个匹配的节点")
        if matching_nodes:
            logger.info(f"匹配的节点索引: {sorted(matching_nodes)}")
        else:
            logger.warning("未找到任何匹配的节点，请检查匹配逻辑")
    elif is_last_generation and not llm_features:
        logger.warning("是最后一代但没有LLM特征，跳过匹配")
    elif not is_last_generation:
        logger.debug(f"不是最后一代 (is_last_generation={is_last_generation})，跳过匹配")

    # 格式化ridge_formula，保留四位小数，并处理化学式下标
    formatted_formula = None
    if ridge_formula:
        # 提取公式中的数值并格式化为四位小数
        # 只格式化系数（在 * 之前）和截距（在 + 或 - 之后），避免匹配特征名称中的数字
        formatted_formula = ridge_formula
        
        # 匹配系数：数字（可能带负号）后跟空格和*，例如 "-31.686514 *"
        def format_coefficient(match):
            num_str = match.group(1)
            try:
                num = float(num_str)
                return f"{num:.4f} ×"  # 使用乘号（×）而不是星号（*），与baseline模型一致
            except ValueError:
                return match.group(0)
        
        formatted_formula = re.sub(r'(-?\d+\.?\d*)\s*\*', format_coefficient, formatted_formula)
        
        # 匹配截距：+ 或 - 后跟数字（在公式末尾），例如 "+ 2687.800795"
        def format_intercept(match):
            sign = match.group(1)
            num_str = match.group(2)
            try:
                num = float(num_str)
                return f"{sign}{num:.4f}"
            except ValueError:
                return match.group(0)
        
        # 只匹配公式末尾的截距（后面没有其他内容，或者只有空格）
        formatted_formula = re.sub(r'([+-])\s*(\d+\.?\d*)\s*$', format_intercept, formatted_formula)
        
        # 处理公式中的特征名称，应用化学式格式化（如 Fe2O3 -> Fe<sub>2</sub>O<sub>3</sub>）
        # 先处理目标变量名称中的化学式
        if target_name:
            formatted_target = format_chemical_formula(target_name)
            # 替换公式开头的目标变量名称
            if formatted_formula.startswith(target_name + " ="):
                formatted_formula = formatted_formula.replace(target_name + " =", formatted_target + " =", 1)
        
        # 处理括号内的特征表达式，例如 (Mean(CaO, Fe2O3))
        # 匹配括号内的内容，但需要避免匹配已经处理过的部分
        def format_feature_in_formula(match):
            full_match = match.group(0)  # 完整的匹配，包括括号
            inner_content = match.group(1)  # 括号内的内容
            
            # 对括号内的每个特征名称应用化学式格式化
            # 匹配特征名称：大写字母开头，可能包含小写字母和数字，如 CaO, Fe2O3, SiO2, Al2O3 等
            def format_single_feature(m):
                feat_name = m.group(0)
                # 检查是否是函数名（如 Mean, Add, Sub 等），如果是则不格式化
                if feat_name in ['Mean', 'Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'Sqrt', 'Squ', 'Ln', 'Log']:
                    return feat_name
                # 否则格式化特征名称
                return format_chemical_formula(feat_name)
            
            # 匹配特征名称或函数名（大写字母开头，可能包含小写字母和数字）
            # 使用单词边界来确保正确匹配
            formatted_expr = re.sub(r'\b([A-Z][a-z]?\d*[A-Z]?\d*)\b', format_single_feature, inner_content)
            return f"({formatted_expr})"
        
        # 匹配括号内的特征表达式，例如 (Mean(CaO, Fe2O3))
        formatted_formula = re.sub(r'\(([^)]+)\)', format_feature_in_formula, formatted_formula)
    
    if HAS_PYGRAPHVIZ:
        # 创建空的有向图对象
        graph = pgv.AGraph(directed=True)
        graph.graph_attr['rankdir'] = 'TB'  # 从上到下布局

        # 添加节点和边
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        # 将标签设置为节点的标签，并指定节点的颜色
        for node, label in labels.items():
            is_matching = node in matching_nodes
            if isinstance(tree[node], gp.Terminal):
                node_obj = graph.get_node(node)
                # 检查label是否是特征名称（IN格式或直接的特征名称）
                formatted_label = label
                if isinstance(label, str):
                    # 检查是否是IN格式（如IN0, IN1等）
                    match = re.match(r'IN(\d+)', label)
                    if match:
                        idx = int(match.group(1))
                        if idx < len(feature_names):
                            # 映射到特征名称并格式化
                            feat_name = feature_names[idx]
                            formatted_label = format_chemical_formula(feat_name)
                        else:
                            formatted_label = label
                    elif label in feature_names:
                        # 直接是特征名称，格式化
                        formatted_label = format_chemical_formula(label)
                    # 如果是数字，保持原样
                    elif isinstance(tree[node].value, (int, float)):
                        formatted_label = label
                
                # 如果包含HTML标签，需要使用HTML格式
                if isinstance(formatted_label, str) and '<sub>' in formatted_label:
                    node_obj.attr['label'] = f'<{formatted_label}>'
                else:
                    node_obj.attr['label'] = str(formatted_label)
                node_obj.attr['style'] = 'filled'
                # 如果匹配LLM特征，使用LLM颜色，否则使用默认颜色
                node_obj.attr['fillcolor'] = LLM_TERMINAL_COLOR if is_matching else TERMINAL_COLOR
            else:
                node_obj = graph.get_node(node)
                node_obj.attr['label'] = label
                node_obj.attr['style'] = 'filled'
                # 如果匹配LLM特征，使用LLM颜色，否则使用默认颜色
                node_obj.attr['fillcolor'] = LLM_FUNCTION_COLOR if is_matching else FUNCTION_COLOR

        # 如果提供了公式，添加公式节点（独立显示在最下面，使用方框和粉色背景）
        if formatted_formula:
            formula_id = "formula"
            # pygraphviz支持HTML标签：如果formula_str包含HTML标签（如<sub>），需要包装在<...>中
            if '<sub>' in formatted_formula or '<SUP>' in formatted_formula.upper():
                # 已经包含HTML标签，直接使用
                html_formula = f'<{formatted_formula}>'
            else:
                # 没有HTML标签，直接使用原字符串
                html_formula = formatted_formula
            graph.add_node(formula_id, label=html_formula, shape='box', style='filled', fillcolor='#FFB6C1')  # 粉色背景
            
            # 使用不可见边来确保公式节点在最下方，并水平对齐
            # 连接到树的根节点和叶子节点
            if nodes:
                # 找到根节点（没有入边的节点，即没有其他节点指向它）
                root_nodes = [n for n in nodes if not any(v == n for u, v in edges)]
                # 找到叶子节点（没有出边的节点，即它不指向任何其他节点）
                leaf_nodes = [n for n in nodes if not any(u == n for u, v in edges)]
                # 连接到第一个根节点和第一个叶子节点（如果存在）
                if root_nodes:
                    first_node = root_nodes[0]
                    graph.add_edge(first_node, formula_id, style='invis')
                if leaf_nodes:
                    last_node = leaf_nodes[0]  # 使用第一个叶子节点
                    graph.add_edge(last_node, formula_id, style='invis')
                elif nodes:
                    # 如果没有叶子节点，使用第一个节点
                    first_node = list(nodes)[0]
                    graph.add_edge(first_node, formula_id, style='invis')

        # 使用随机种子与目标作为文件名
        suffix = f"_{target_name}" if target_name else ""
        
        # 先生成不带公式的PDF（原来的版本）
        # 创建不带公式的图
        graph_no_formula = pgv.AGraph(directed=True)
        graph_no_formula.graph_attr['rankdir'] = 'TB'
        graph_no_formula.add_nodes_from(nodes)
        graph_no_formula.add_edges_from(edges)
        
        # 设置节点标签和颜色（与之前相同）
        for node, label in labels.items():
            is_matching = node in matching_nodes
            if isinstance(tree[node], gp.Terminal):
                node_obj = graph_no_formula.get_node(node)
                formatted_label = label
                if isinstance(label, str):
                    match = re.match(r'IN(\d+)', label)
                    if match:
                        idx = int(match.group(1))
                        if idx < len(feature_names):
                            feat_name = feature_names[idx]
                            formatted_label = format_chemical_formula(feat_name)
                        else:
                            formatted_label = label
                    elif label in feature_names:
                        formatted_label = format_chemical_formula(label)
                    elif isinstance(tree[node].value, (int, float)):
                        formatted_label = label
                
                if isinstance(formatted_label, str) and '<sub>' in formatted_label:
                    node_obj.attr['label'] = f'<{formatted_label}>'
                else:
                    node_obj.attr['label'] = str(formatted_label)
                node_obj.attr['style'] = 'filled'
                node_obj.attr['fillcolor'] = LLM_TERMINAL_COLOR if is_matching else TERMINAL_COLOR
            else:
                node_obj = graph_no_formula.get_node(node)
                node_obj.attr['label'] = label
                node_obj.attr['style'] = 'filled'
                node_obj.attr['fillcolor'] = LLM_FUNCTION_COLOR if is_matching else FUNCTION_COLOR
        
        graph_no_formula.layout(prog='dot')
        filename = f"{randomSeeds}{suffix}.pdf"
        filepath = os.path.join(tree_save_fileplace, filename)
        graph_no_formula.draw(filepath)
        
        # 如果有公式，生成带公式的PDF
        if formatted_formula:
            graph.layout(prog='dot')
            filename_with_formula = f"{randomSeeds}{suffix}_with_formula.pdf"
            filepath_with_formula = os.path.join(tree_save_fileplace, filename_with_formula)
            graph.draw(filepath_with_formula)
    elif HAS_GRAPHVIZ:
        # 使用 graphviz 生成 PDF
        g = _GraphvizDigraph(format='pdf')
        g.attr('graph', rankdir='TB')  # 从上到下布局
        
        for node in nodes:
            label = labels.get(node, str(node))
            # 终端与函数节点使用不同填充色
            is_terminal = isinstance(tree[node], gp.Terminal)
            is_matching = node in matching_nodes
            
            # 格式化标签（如果是特征名称）
            formatted_label = label
            if is_terminal and isinstance(label, str):
                # 检查是否是IN格式（如IN0, IN1等）
                match = re.match(r'IN(\d+)', label)
                if match:
                    idx = int(match.group(1))
                    if idx < len(feature_names):
                        # 映射到特征名称并格式化
                        feat_name = feature_names[idx]
                        formatted_label = format_chemical_formula(feat_name)
                    else:
                        formatted_label = label
                elif label in feature_names:
                    # 直接是特征名称，格式化
                    formatted_label = format_chemical_formula(label)
                # 如果是数字，保持原样
                elif isinstance(tree[node].value, (int, float)):
                    formatted_label = label
            
            # 如果匹配LLM特征，使用LLM颜色，否则使用默认颜色
            if is_terminal:
                fillcolor = LLM_TERMINAL_COLOR if is_matching else TERMINAL_COLOR
            else:
                fillcolor = LLM_FUNCTION_COLOR if is_matching else FUNCTION_COLOR
            
            # 如果包含HTML标签，需要使用HTML格式
            if isinstance(formatted_label, str) and '<sub>' in formatted_label:
                g.node(str(node), label=f'<{formatted_label}>', style='filled', fillcolor=fillcolor)
            else:
                g.node(str(node), label=str(formatted_label), style='filled', fillcolor=fillcolor)
        for u, v in edges:
            g.edge(str(u), str(v))
        
        # 如果提供了公式，添加公式节点（独立显示在最下面，使用方框和粉色背景）
        if formatted_formula:
            formula_id = "formula"
            # Graphviz支持HTML标签：如果formula_str包含HTML标签（如<sub>），需要包装在<...>中
            if '<sub>' in formatted_formula or '<SUP>' in formatted_formula.upper():
                # 已经包含HTML标签，直接使用
                html_formula = f'<{formatted_formula}>'
            else:
                # 没有HTML标签，直接使用原字符串
                html_formula = formatted_formula
            g.node(formula_id, html_formula, shape='box', style='filled', fillcolor='#FFB6C1')  # 粉色背景
            
            # 使用rank='sink'来确保公式节点在最下方
            with g.subgraph() as sink_subgraph:
                sink_subgraph.attr(rank='sink')
                sink_subgraph.node(formula_id)
            
            # 使用不可见边确保公式节点在所有节点下方，并水平对齐
            # 连接到树的根节点和叶子节点
            if nodes:
                # 找到根节点（没有入边的节点，即没有其他节点指向它）
                root_nodes = [n for n in nodes if not any(v == n for u, v in edges)]
                # 找到叶子节点（没有出边的节点，即它不指向任何其他节点）
                leaf_nodes = [n for n in nodes if not any(u == n for u, v in edges)]
                # 连接到第一个根节点和第一个叶子节点（如果存在）
                if root_nodes:
                    first_node = str(root_nodes[0])
                    g.edge(first_node, formula_id, style='invis', weight='0', constraint='false')
                if leaf_nodes:
                    last_node = str(leaf_nodes[0])  # 使用第一个叶子节点
                    g.edge(last_node, formula_id, style='invis', weight='0', constraint='false')
                elif nodes:
                    # 如果没有叶子节点，使用第一个节点
                    first_node = str(list(nodes)[0])
                    g.edge(first_node, formula_id, style='invis', weight='0', constraint='false')
        
        suffix = f"_{target_name}" if target_name else ""
        
        # 生成不带公式的PDF（原来的版本）
        filename = f"{randomSeeds}{suffix}"
        filepath = os.path.join(tree_save_fileplace, filename)
        # 创建不带公式的图副本
        g_no_formula = _GraphvizDigraph(format='pdf')
        g_no_formula.attr('graph', rankdir='TB')
        # 复制所有节点和边（除了公式节点）
        for node in nodes:
            label = labels.get(node, str(node))
            is_terminal = isinstance(tree[node], gp.Terminal)
            is_matching = node in matching_nodes
            
            formatted_label = label
            if is_terminal and isinstance(label, str):
                match = re.match(r'IN(\d+)', label)
                if match:
                    idx = int(match.group(1))
                    if idx < len(feature_names):
                        feat_name = feature_names[idx]
                        formatted_label = format_chemical_formula(feat_name)
                    else:
                        formatted_label = label
                elif label in feature_names:
                    formatted_label = format_chemical_formula(label)
                elif isinstance(tree[node].value, (int, float)):
                    formatted_label = label
            
            if is_terminal:
                fillcolor = LLM_TERMINAL_COLOR if is_matching else TERMINAL_COLOR
            else:
                fillcolor = LLM_FUNCTION_COLOR if is_matching else FUNCTION_COLOR
            
            if isinstance(formatted_label, str) and '<sub>' in formatted_label:
                g_no_formula.node(str(node), label=f'<{formatted_label}>', style='filled', fillcolor=fillcolor)
            else:
                g_no_formula.node(str(node), label=str(formatted_label), style='filled', fillcolor=fillcolor)
        for u, v in edges:
            g_no_formula.edge(str(u), str(v))
        
        g_no_formula.render(filepath, cleanup=True)  # 生成 filepath.pdf
        filepath_no_formula = f"{filepath}.pdf"
        
        # 如果有公式，生成带公式的PDF
        if formatted_formula:
            filename_with_formula = f"{randomSeeds}{suffix}_with_formula"
            filepath_with_formula = os.path.join(tree_save_fileplace, filename_with_formula)
            g.render(filepath_with_formula, cleanup=True)  # 生成 filepath_with_formula.pdf
            filepath = f"{filepath_with_formula}.pdf"
        else:
            filepath = filepath_no_formula
    else:
        raise RuntimeError("Neither pygraphviz nor graphviz is available to render PDF.")

    column_sums = [sum(col) for col in zip(*feature_count_list)]
    logger = logging.getLogger(__name__)
    logger.info("randomSeeds = {}时,Hof 已存入；Best 全体统计特征总和为：{},".format(randomSeeds, column_sums))

