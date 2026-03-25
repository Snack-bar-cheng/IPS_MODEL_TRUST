"""
特征配置组件模块
"""
import streamlit as st
import pandas as pd


def render_feature_config(df):
    """
    渲染特征配置部分
    
    返回:
        target_column: 目标变量列名
        exclude_columns: 排除的列列表
        num_features: 生成特征数量
        feature_columns: 计算后的特征列列表
    """
    st.subheader("2. 特征配置")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 选择目标变量
        # 如果session_state中有预设值，使用预设值
        default_target_index = 0
        if 'target_column_selectbox' in st.session_state and st.session_state.target_column_selectbox in df.columns.tolist():
            default_target_index = df.columns.tolist().index(st.session_state.target_column_selectbox)
        
        target_column = st.selectbox(
            "选择目标变量",
            options=df.columns.tolist(),
            index=default_target_index,
            help="选择要预测的目标变量",
            key="target_column_selectbox"
        )
    
    with col2:
        # 删除列（多选）
        # 如果session_state中有预设值，使用预设值
        exclude_options = [col for col in df.columns if col != target_column]
        default_exclude = []
        if 'exclude_columns_multiselect' in st.session_state:
            default_exclude = [col for col in st.session_state.exclude_columns_multiselect if col in exclude_options]
        
        exclude_columns = st.multiselect(
            "删除列（排除这些列）",
            options=exclude_options,
            default=default_exclude,
            help="选择要排除的列，其余列将作为特征列",
            key="exclude_columns_multiselect"
        )
    
    with col3:
        # 生成数量
        num_features = st.number_input(
            "生成特征数量",
            min_value=1,
            max_value=1000,
            value=10,
            help="要生成的LLM特征数量",
            key="num_features_input"
        )
    
    # 表达式高度限制（新增，采用DEAP标准）
    max_height = st.number_input(
        "表达式高度限制",
        min_value=0,
        max_value=10,
        value=1,
        help="生成的表达式树的高度限制（采用DEAP标准）。高度计算规则：叶子节点（特征名）height为0，内部节点：height = 1 + max(所有操作数的height)。例如：A+B height为1，max(A+B, C/D) height为2。生成的表达式必须恰好等于此高度，不能多也不能少。",
        key="max_height_input"
    )
    
    # 计算特征列（排除目标变量和删除列）
    feature_columns = [col for col in df.columns if col != target_column and col not in exclude_columns]
    
    if len(feature_columns) == 0:
        st.warning("⚠️ 没有可用的特征列，请检查删除列设置")
    else:
        st.info(f"✅ 将使用 {len(feature_columns)} 个特征列: {', '.join(feature_columns[:10])}{'...' if len(feature_columns) > 10 else ''}")
    
    return target_column, exclude_columns, num_features, feature_columns, max_height

