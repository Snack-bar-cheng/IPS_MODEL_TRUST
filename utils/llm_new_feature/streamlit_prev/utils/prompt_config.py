"""
Prompt配置组件模块
"""
import os
import importlib
import streamlit as st
import re
import llm_new_feature

from .pdf_extract import extract_text_from_pdf_bytes

# llm_prompt.py 磁盘路径，用于检测模板变更并 importlib.reload，避免 Streamlit 缓存旧模块
_LLM_PROMPT_PATH = os.path.join(os.path.dirname(llm_new_feature.__file__), "llm_prompt.py")


def _sync_llm_prompt_module() -> bool:
    """
    若 llm_prompt.py 已保存则重新加载模块。
    返回 True 表示「相对上次运行模板文件已变化」，用于刷新前端文本框与丢弃旧的手动编辑缓存。
    """
    try:
        mtime = os.path.getmtime(_LLM_PROMPT_PATH)
    except OSError:
        return False
    prev = st.session_state.get("_llm_prompt_mtime")
    if prev == mtime:
        return False
    import llm_new_feature.llm_prompt as lp
    importlib.reload(lp)
    st.session_state._llm_prompt_mtime = mtime
    return prev is not None


def _combine_pdf_and_prompt(pdf_text: str, prompt_text: str) -> str:
    """将 PDF 参考文本与用户可编辑的 Prompt 合并后发给大模型。"""
    pdf_text = (pdf_text or "").strip()
    prompt_text = prompt_text or ""
    if not pdf_text:
        return prompt_text
    return (
        "## 参考资料（来自上传的 PDF）\n\n"
        + pdf_text
        + "\n\n---\n\n"
        + prompt_text
    )


def get_default_prompt(target_name: str, feature_names: list, num_features: int = 10, max_height: int = None) -> str:
    """获取默认的完整prompt（始终从当前 llm_prompt 模块读取，避免缓存旧模板）"""
    import llm_new_feature.llm_prompt as lp
    return lp.build_batch_feature_generation_prompt(
        target_name, feature_names, num_features, None, None, max_height
    )


def check_config_changed(target_column, exclude_columns, num_features, max_height=None):
    """检查配置是否改变"""
    config_changed = False
    
    # 检查目标列是否改变
    if target_column != st.session_state.last_target_column:
        config_changed = True
        st.session_state.last_target_column = target_column
    
    # 检查删除列是否改变
    exclude_columns_set = set(exclude_columns)
    last_exclude_columns_set = set(st.session_state.last_exclude_columns or [])
    if exclude_columns_set != last_exclude_columns_set:
        config_changed = True
        st.session_state.last_exclude_columns = exclude_columns
    
    # 检查生成数量是否改变
    if num_features != st.session_state.last_num_features:
        config_changed = True
        st.session_state.last_num_features = num_features
    
    # 检查高度限制是否改变
    if max_height is not None:
        if max_height != st.session_state.get('last_max_height'):
            config_changed = True
            st.session_state.last_max_height = max_height
    
    return config_changed


def update_prompt_on_config_change(target_column, exclude_columns, num_features, avoid_duplicates, max_height=None):
    """
    当配置改变时更新prompt
    
    返回:
        config_changed: 配置是否改变
    """
    config_changed = check_config_changed(target_column, exclude_columns, num_features, max_height)
    
    # 如果配置改变，清除用户编辑的prompt以便重新生成
    if config_changed:
        st.session_state.user_edited_prompt = None
    
    return config_changed


def get_existing_features_info():
    """获取已存在的特征信息（用于避免重复）"""
    existing_features_info = ""
    existing_trees = []
    
    if st.session_state.generated_features:
        # 提取已存在的数学表示
        existing_notations = [
            feature.get('notation', '') 
            for feature in st.session_state.generated_features 
            if feature.get('notation', '').strip()
        ]
        # 提取已存在的tree结构
        existing_trees = [
            feature.get('tree') 
            for feature in st.session_state.generated_features 
            if feature.get('tree')
        ]
        
        if existing_notations:
            existing_features_info = "\n## 已存在的特征（请避免生成相同的特征）\n\n"
            existing_features_info += "已存在的特征数学表示列表：\n"
            for i, notation in enumerate(existing_notations, 1):
                existing_features_info += f"{i}. {notation}\n"
            if len(existing_notations) > 10:
                existing_features_info += f"... 还有 {len(existing_notations) - 10} 个已存在的特征\n"
    
    return existing_features_info, existing_trees


def build_prompt_with_existing_features(base_prompt, existing_features_info):
    """将已存在的特征信息添加到prompt中"""
    if not existing_features_info:
        return base_prompt
    
    # 在prompt中查找插入位置（在"## 预测目标"之后）
    if "## 预测目标" in base_prompt:
        parts = base_prompt.split("## 预测目标", 1)
        if len(parts) == 2:
            # 在预测目标部分之后插入已存在特征信息
            target_section = parts[1]
            # 查找下一个##标题的位置
            next_section_match = re.search(r'\n## ', target_section)
            if next_section_match:
                insert_pos = next_section_match.start()
                base_prompt = parts[0] + "## 预测目标" + target_section[:insert_pos] + existing_features_info + target_section[insert_pos:]
            else:
                # 如果没有下一个标题，直接追加
                base_prompt = base_prompt + existing_features_info
    else:
        # 如果没有找到预测目标部分，直接追加
        base_prompt = base_prompt + existing_features_info
    
    return base_prompt


def render_prompt_config(target_column, exclude_columns, feature_columns, num_features, max_height=None):
    """
    渲染Prompt配置部分
    
    参数:
        target_column: 目标变量列名
        exclude_columns: 排除的列列表
        feature_columns: 计算后的特征列列表
        num_features: 生成特征数量
    
    返回:
        task_context: 最终的prompt文本
        avoid_duplicates: 是否避免重复
    """
    st.subheader("3. Prompt配置")

    # 若磁盘上的 llm_prompt.py 已更新：reload 模块、丢弃旧的手动编辑缓存、刷新 text_area
    if _sync_llm_prompt_module():
        st.session_state.user_edited_prompt = None
        st.session_state.prompt_update_counter = st.session_state.get("prompt_update_counter", 0) + 1

    # 避免重复特征的勾选框
    avoid_duplicates = st.checkbox(
        "避免生成重复特征（自动读取表格中已存在的数学表示）",
        value=st.session_state.avoid_duplicates_state,
        help="勾选后，会自动将当前表格中已存在的特征数学表示添加到prompt中，避免生成重复的表达式",
        key="avoid_duplicates_checkbox"
    )
    
    # 检测勾选框状态是否改变
    checkbox_changed = avoid_duplicates != st.session_state.avoid_duplicates_state
    if checkbox_changed:
        st.session_state.avoid_duplicates_state = avoid_duplicates
        st.session_state.user_edited_prompt = None
    
    # 检查配置是否改变（目标列、删除列、生成数量、高度限制）
    config_changed = update_prompt_on_config_change(
        target_column, exclude_columns, num_features, avoid_duplicates, max_height
    )
    
    # 获取已存在的特征信息（用于避免重复）
    existing_features_info = ""
    if avoid_duplicates:
        existing_features_info, _ = get_existing_features_info()
    
    # 获取基础prompt（使用用户设置的特征数量和高度限制）
    base_prompt = get_default_prompt(target_column, feature_columns, num_features, max_height) if len(feature_columns) > 0 else ""
    
    # 如果启用了避免重复，将已存在的特征信息添加到prompt中
    if avoid_duplicates and existing_features_info:
        base_prompt = build_prompt_with_existing_features(base_prompt, existing_features_info)
    
    # 确定要显示的prompt值
    # 如果配置或勾选框状态改变，使用新生成的prompt
    # 如果用户之前编辑过prompt且状态未改变，使用用户编辑的版本
    if checkbox_changed or config_changed:
        # 配置改变时，强制使用新的base_prompt
        display_prompt = base_prompt
        st.session_state.user_edited_prompt = None
    elif st.session_state.user_edited_prompt is not None:
        # 使用用户编辑的版本
        display_prompt = st.session_state.user_edited_prompt
    else:
        # 使用基础prompt
        display_prompt = base_prompt
    
    # 如果配置改变，增加计数器以强制更新text_area（使用动态key）
    if config_changed or checkbox_changed:
        st.session_state.prompt_update_counter += 1

    # ---------- PDF 参考资料（合并进发给大模型的内容，显示在「完整 Prompt」上方）----------
    st.markdown("**PDF 参考资料（可选）**")
    uploaded_pdf = st.file_uploader(
        "上传 PDF，文本将一并提供给大模型分析",
        type=["pdf"],
        help="支持多页 PDF；解析后的纯文本会附加在下方「完整 Prompt」之前发送。扫描版 PDF 若无文字层则可能无法提取。",
        key="pdf_knowledge_uploader",
    )
    if uploaded_pdf is not None:
        sig = (uploaded_pdf.name, uploaded_pdf.size)
        if st.session_state.pdf_parse_signature != sig:
            raw = uploaded_pdf.getvalue()
            extracted, err = extract_text_from_pdf_bytes(raw)
            if err:
                st.error(err)
                st.session_state.pdf_knowledge_text = ""
                st.session_state.pdf_knowledge_name = None
                st.session_state.pdf_parse_signature = None
            else:
                st.session_state.pdf_knowledge_text = extracted
                st.session_state.pdf_knowledge_name = uploaded_pdf.name
                st.session_state.pdf_parse_signature = sig
                st.success(f"已解析 PDF：{uploaded_pdf.name}（约 {len(extracted)} 字符）")

    if st.session_state.pdf_knowledge_text:
        meta = st.session_state.pdf_knowledge_name or "已加载"
        st.caption(f"当前将随请求发送的 PDF 参考：{meta}（{len(st.session_state.pdf_knowledge_text)} 字符）")
        with st.expander("预览 PDF 解析文本（前 3000 字）", expanded=False):
            preview = st.session_state.pdf_knowledge_text[:3000]
            st.text(preview + ("…" if len(st.session_state.pdf_knowledge_text) > 3000 else ""))
        if st.button("清除 PDF 参考内容", key="clear_pdf_knowledge"):
            st.session_state.pdf_knowledge_text = ""
            st.session_state.pdf_knowledge_name = None
            st.session_state.pdf_parse_signature = None
            if "pdf_knowledge_uploader" in st.session_state:
                del st.session_state["pdf_knowledge_uploader"]
            st.rerun()
    
    # 显示完整prompt并允许编辑
    # 使用动态key确保配置改变时能够更新
    task_context = st.text_area(
        "完整Prompt（可编辑）",
        value=display_prompt,
        help="完整的LLM prompt，可以编辑。当选择目标列、删除列或勾选避免重复特征后，会自动更新此prompt。若上传了 PDF，其文本会附加在本段内容之前一并发送。",
        height=400,
        key=f"prompt_text_area_{st.session_state.prompt_update_counter}"
    )
    
    # 保存用户编辑的prompt（如果用户修改了prompt）
    if task_context != base_prompt:
        st.session_state.user_edited_prompt = task_context

    # 发给后端的 task_context：PDF 参考 + 用户 Prompt
    full_task_context = _combine_pdf_and_prompt(st.session_state.pdf_knowledge_text, task_context)
    return full_task_context, avoid_duplicates

