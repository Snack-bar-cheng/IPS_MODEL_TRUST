"""
侧边栏组件模块
"""
import streamlit as st
import pandas as pd
import json
import logging
import os

logger = logging.getLogger(__name__)


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("配置")
        
        # 数据集上传
        st.subheader("1. 上传数据集")
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="上传包含特征列的数据集文件"
        )
        
        df = None
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"数据集加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")
            except Exception as e:
                st.error(f"读取数据集失败: {e}")
                df = None
        
        st.markdown("---")
        
        # JSON特征文件上传
        st.subheader("📁 加载JSON特征文件")
        
        # 添加加载方式选择（默认选择"上传本地文件"）
        # 初始化默认值
        if 'json_load_method' not in st.session_state:
            st.session_state.json_load_method = "上传本地文件"
        
        load_method = st.radio(
            "选择加载方式",
            ["上传本地文件", "从服务器目录选择"],
            index=0 if st.session_state.json_load_method == "上传本地文件" else 1,
            help="可以选择从服务器json_save目录加载或上传本地JSON文件",
            key="json_load_method"
        )
        
        uploaded_json = None
        selected_server_file = None
        
        if load_method == "从服务器目录选择":
            # 获取json_save目录路径
            # sidebar.py 位于 streamlit_prev/utils/，需要回到 llm_new_feature 目录
            current_dir = os.path.dirname(os.path.abspath(__file__))  # streamlit_prev/utils/
            parent_dir = os.path.dirname(current_dir)  # streamlit_prev/
            base_dir = os.path.dirname(parent_dir)  # llm_new_feature/
            json_save_dir = os.path.join(base_dir, 'json_save')  # llm_new_feature/json_save
            
            # 列出目录中的所有JSON文件
            json_files = []
            if os.path.exists(json_save_dir):
                json_files = sorted([f for f in os.listdir(json_save_dir) if f.endswith('.json')], reverse=True)
            
            if json_files:
                # 添加一个占位符选项作为默认值（默认不加载）
                options = ["（请选择文件...）"] + json_files
                
                # 如果之前选择了文件，保持选择；否则默认选择占位符
                if 'server_json_selector' not in st.session_state or st.session_state.server_json_selector not in options:
                    st.session_state.server_json_selector = options[0]  # 默认选择占位符
                
                selected_option = st.selectbox(
                    f"选择JSON文件（找到 {len(json_files)} 个文件）",
                    options,
                    index=options.index(st.session_state.server_json_selector) if st.session_state.server_json_selector in options else 0,
                    help="从服务器json_save目录中选择要加载的JSON文件，选择后点击下方的加载按钮",
                    key="server_json_selector"
                )
                
                # 只有当选择了实际文件（不是占位符）时才显示加载按钮
                if selected_option and selected_option != "（请选择文件...）":
                    selected_server_file = selected_option
                    file_path = os.path.join(json_save_dir, selected_server_file)
                    
                    # 添加加载按钮
                    if st.button("📂 加载选中的文件", key="load_server_json", type="primary"):
                        # 检查是否是新的文件
                        is_new_file = selected_server_file != st.session_state.last_uploaded_json_name
                        
                        if is_new_file:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    json_data = json.load(f)
                                
                                # 检查JSON格式
                                if 'features' not in json_data:
                                    st.error("❌ JSON文件格式错误：缺少 'features' 字段")
                                else:
                                    features = json_data.get('features', [])
                                    if len(features) > 0:
                                        # 加载特征到session state
                                        st.session_state.generated_features = features
                                        
                                        # 如果JSON中有target_name和feature_names，也加载它们
                                        if 'target_name' in json_data:
                                            st.session_state.target_name = json_data['target_name']
                                        if 'feature_names' in json_data:
                                            st.session_state.feature_names = json_data['feature_names']
                                        
                                        # 尝试从第一个特征中获取model_name（如果存在）
                                        if features and 'model_name' in features[0]:
                                            st.session_state.model_name = features[0].get('model_name')
                                        
                                        # 更新最后加载的文件名和文件路径
                                        st.session_state.last_uploaded_json_name = selected_server_file
                                        st.session_state.loaded_json_file_path = file_path  # 保存完整路径用于追加特征和保存备份
                                        st.session_state.enable_append_mode = True  # 启用追加模式
                                        
                                        st.success(f"✅ 成功加载 {len(features)} 个特征！")
                                        st.info(f"📊 目标变量: {json_data.get('target_name', 'N/A')}")
                                        st.info(f"🔢 特征数量: {len(features)}")
                                        
                                        # 自动刷新页面以显示特征
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ JSON文件中的features字段为空")
                            except json.JSONDecodeError as e:
                                st.error(f"❌ JSON解析失败: {e}")
                            except Exception as e:
                                st.error(f"❌ 读取JSON文件失败: {e}")
                                logger.exception("读取JSON文件时出错")
                        else:
                            # 显示已加载的信息
                            if st.session_state.generated_features:
                                st.info(f"📁 文件 {selected_server_file} 已经加载过了")
                                st.success(f"✅ 当前已加载 {len(st.session_state.generated_features)} 个特征")
                                st.info(f"📊 目标变量: {st.session_state.target_name or 'N/A'}")
                else:
                    # 显示提示信息（选择了占位符）
                    st.info("💡 请从下拉列表中选择一个JSON文件，然后点击加载按钮")
            else:
                st.warning(f"⚠️ 在目录 {json_save_dir} 中未找到JSON文件")
        else:
            # 上传本地文件的方式（原有逻辑）
            uploaded_json = st.file_uploader(
                "选择JSON文件",
                type=['json'],
                help="上传包含特征定义的JSON文件（如 llm_Ash_Deformation_*.json）",
                key="json_uploader"
            )
        
        if uploaded_json is not None:
            # 检查是否是新的文件（通过文件名判断）
            current_file_name = uploaded_json.name
            is_new_file = current_file_name != st.session_state.last_uploaded_json_name
            
            if is_new_file:
                try:
                    # 重置文件指针（因为可能已经被读取过）
                    uploaded_json.seek(0)
                    # 读取JSON文件
                    json_data = json.load(uploaded_json)
                    
                    # 检查JSON格式
                    if 'features' not in json_data:
                        st.error("❌ JSON文件格式错误：缺少 'features' 字段")
                    else:
                        features = json_data.get('features', [])
                        if len(features) > 0:
                            # 加载特征到session state
                            st.session_state.generated_features = features
                            
                            # 如果JSON中有target_name和feature_names，也加载它们
                            # 注意：model_name已经包含在每个feature对象中，不需要从顶层加载
                            if 'target_name' in json_data:
                                st.session_state.target_name = json_data['target_name']
                            if 'feature_names' in json_data:
                                st.session_state.feature_names = json_data['feature_names']
                            
                            # 尝试从第一个特征中获取model_name（如果存在）
                            if features and 'model_name' in features[0]:
                                st.session_state.model_name = features[0].get('model_name')
                            
                            # 更新最后上传的文件名
                            st.session_state.last_uploaded_json_name = current_file_name
                            # 对于上传的文件，无法获取服务器路径，设置为None（追加模式将不可用）
                            st.session_state.loaded_json_file_path = None
                            st.session_state.enable_append_mode = False
                            
                            st.success(f"✅ 成功加载 {len(features)} 个特征！")
                            st.info(f"📊 目标变量: {json_data.get('target_name', 'N/A')}")
                            st.info(f"🔢 特征数量: {len(features)}")
                            
                            # 自动刷新页面以显示特征
                            st.rerun()
                        else:
                            st.warning("⚠️ JSON文件中的features字段为空")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON解析失败: {e}")
                except Exception as e:
                    st.error(f"❌ 读取JSON文件失败: {e}")
                    logger.exception("读取JSON文件时出错")
            else:
                # 显示已加载的信息
                if st.session_state.generated_features:
                    st.success(f"✅ 已加载 {len(st.session_state.generated_features)} 个特征")
                    st.info(f"📊 目标变量: {st.session_state.target_name or 'N/A'}")
                    st.info(f"📁 文件: {current_file_name}")
    
    return df

