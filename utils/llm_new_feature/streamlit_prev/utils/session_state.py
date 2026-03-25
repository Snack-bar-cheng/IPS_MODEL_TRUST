"""
Session State管理模块
"""
import streamlit as st


def init_session_state():
    """初始化session state"""
    if 'generated_features' not in st.session_state:
        st.session_state.generated_features = []
    if 'target_name' not in st.session_state:
        st.session_state.target_name = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = []
    if 'last_uploaded_json_name' not in st.session_state:
        st.session_state.last_uploaded_json_name = None
    if 'avoid_duplicates_state' not in st.session_state:
        st.session_state.avoid_duplicates_state = True  # 默认勾选避免生成重复特征
    if 'user_edited_prompt' not in st.session_state:
        st.session_state.user_edited_prompt = None
    # 用于跟踪配置变化，触发prompt更新
    if 'last_target_column' not in st.session_state:
        st.session_state.last_target_column = None
    if 'last_exclude_columns' not in st.session_state:
        st.session_state.last_exclude_columns = None
    if 'last_num_features' not in st.session_state:
        st.session_state.last_num_features = None
    if 'prompt_update_counter' not in st.session_state:
        st.session_state.prompt_update_counter = 0
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    # 自动重复生成相关状态
    if 'auto_repeat_mode' not in st.session_state:
        st.session_state.auto_repeat_mode = False  # 是否处于自动重复模式
    if 'auto_repeat_trigger' not in st.session_state:
        st.session_state.auto_repeat_trigger = False  # 是否触发自动重复生成
    if 'auto_repeat_waiting' not in st.session_state:
        st.session_state.auto_repeat_waiting = False  # 是否正在等待中
    if 'auto_repeat_count' not in st.session_state:
        st.session_state.auto_repeat_count = 0  # 自动重复生成的次数计数器
    if 'auto_repeat_wait_until' not in st.session_state:
        st.session_state.auto_repeat_wait_until = None  # 等待到的时间戳
    if 'auto_repeat_total_times' not in st.session_state:
        st.session_state.auto_repeat_total_times = 2  # 自动重复的总次数（默认2次）
    # LLM原始响应相关状态
    if 'llm_raw_responses' not in st.session_state:
        st.session_state.llm_raw_responses = []  # 存储所有LLM原始响应的列表
    if 'show_raw_response' not in st.session_state:
        st.session_state.show_raw_response = False  # 是否显示原始响应
    # 追加特征相关状态
    if 'loaded_json_file_path' not in st.session_state:
        st.session_state.loaded_json_file_path = None  # 保存加载的JSON文件完整路径（用于追加特征和保存时备份）
    if 'enable_append_mode' not in st.session_state:
        st.session_state.enable_append_mode = False  # 是否处于追加特征模式
    if 'append_features_triggered' not in st.session_state:
        st.session_state.append_features_triggered = False  # 是否触发了追加特征按钮
    if 'append_config_set' not in st.session_state:
        st.session_state.append_config_set = False  # 是否已经设置了追加特征配置
    # PDF 参考资料（解析后的纯文本，供与大模型合并）
    if 'pdf_knowledge_text' not in st.session_state:
        st.session_state.pdf_knowledge_text = ""
    if 'pdf_knowledge_name' not in st.session_state:
        st.session_state.pdf_knowledge_name = None
    if 'pdf_parse_signature' not in st.session_state:
        st.session_state.pdf_parse_signature = None

