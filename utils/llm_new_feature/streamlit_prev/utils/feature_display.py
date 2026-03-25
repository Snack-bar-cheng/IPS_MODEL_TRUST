"""
特征显示和保存组件模块
"""
import streamlit as st
import pandas as pd
import os
from llm_new_feature.feature_generator import save_features_to_json


def render_feature_display():
    """渲染特征显示和保存部分"""
    if not st.session_state.generated_features:
        return
    
    st.subheader("📋 特征预览与编辑")
    
    # 追加特征按钮（只有当从服务器目录加载JSON文件时才显示）
    if st.session_state.get('loaded_json_file_path') and st.session_state.get('enable_append_mode'):
        col_append = st.columns([1, 4])
        with col_append[0]:
            if st.button("➕ 追加特征", type="primary", use_container_width=True, key="append_features_btn"):
                # 设置追加模式标记，用于触发配置自动设置
                st.session_state.append_features_triggered = True
                st.rerun()
    
    # 创建DataFrame用于显示
    features_df_data = []
    for i, feature in enumerate(st.session_state.generated_features):
        # 如果特征中没有model_name，尝试从session_state获取，或者使用N/A
        model_name = feature.get('model_name') or st.session_state.get('model_name') or 'N/A'
        height = feature.get('height', feature.get('depth', 'N/A'))  # 获取高度信息（兼容旧字段名）
        features_df_data.append({
            '序号': i + 1,
            '模型名称': model_name,
            '高度': height,
            '数学表示': feature.get('notation', 'N/A'),
            '描述': feature.get('description', 'N/A')
        })
    
    features_df = pd.DataFrame(features_df_data)
    
    # 显示表格（使用data_editor支持删除）
    edited_df = st.data_editor(
        features_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            '序号': st.column_config.NumberColumn('序号', disabled=True),
            '模型名称': st.column_config.TextColumn('模型名称', disabled=True),
            '高度': st.column_config.NumberColumn('高度', disabled=True, format="%d"),
            '数学表示': st.column_config.TextColumn('数学表示', disabled=True),
            '描述': st.column_config.TextColumn('描述', disabled=True)
        },
        key="features_editor"
    )
    
    # 检查是否有行被删除
    if len(edited_df) < len(features_df):
        # 用户删除了行，更新session state
        remaining_indices = [int(idx) - 1 for idx in edited_df['序号'].tolist()]
        st.session_state.generated_features = [
            st.session_state.generated_features[idx] 
            for idx in remaining_indices
        ]
        st.rerun()
    
    # 保存按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        save_button_label = "💾 保存为JSON文件"
        if st.session_state.get('loaded_json_file_path') and st.session_state.get('enable_append_mode'):
            save_button_label = "💾 保存并覆盖原文件"
        
        if st.button(save_button_label, type="primary", use_container_width=True):
            if len(st.session_state.generated_features) > 0:
                # 使用session state中的值，如果没有则使用默认值
                target_name = st.session_state.target_name or "Unknown_Target"
                feature_names = st.session_state.feature_names or []
                
                # 检查是否是从服务器目录加载的文件（需要备份并覆盖）
                if st.session_state.get('loaded_json_file_path') and st.session_state.get('enable_append_mode'):
                    original_file_path = st.session_state.loaded_json_file_path
                    original_dir = os.path.dirname(original_file_path)
                    original_filename = os.path.basename(original_file_path)
                    
                    # 将原文件重命名为backup前缀
                    backup_filename = f"backup_{original_filename}"
                    backup_file_path = os.path.join(original_dir, backup_filename)
                    
                    try:
                        # 重命名原文件为备份文件
                        if os.path.exists(original_file_path):
                            os.rename(original_file_path, backup_file_path)
                            st.info(f"📦 原文件已重命名为备份: `{backup_filename}`")
                        else:
                            st.warning(f"⚠️ 原文件不存在: `{original_filename}`")
                    except Exception as e:
                        st.warning(f"⚠️ 重命名原文件失败: {e}")
                    
                    # 生成新的文件名（使用当前时间戳）
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"llm_{target_name}_{timestamp}.json"
                    new_file_path = os.path.join(original_dir, new_filename)
                    
                    # 保存到新文件路径
                    try:
                        import json
                        
                        output_data = {
                            "target_name": target_name,
                            "feature_names": feature_names,
                            "num_features": len(st.session_state.generated_features),
                            "generated_at": timestamp,
                            "features": st.session_state.generated_features
                        }
                        
                        # 使用与save_features_to_json相同的格式
                        json_str = json.dumps(output_data, ensure_ascii=False, indent=2)
                        import re
                        # 将feature_names数组格式化为一行
                        pattern = r'"feature_names":\s*\[\s*(.*?)\s*\]'
                        def replace_feature_names(match):
                            content = match.group(1)
                            content = re.sub(r'\s+', ' ', content.strip())
                            content = re.sub(r',\s*', ', ', content)
                            return f'"feature_names": [{content}]'
                        json_str = re.sub(pattern, replace_feature_names, json_str, flags=re.DOTALL)
                        
                        # 将operands数组格式化为一行
                        def format_simple_operands(match):
                            full_match = match.group(0)
                            if '{' in full_match:
                                return full_match
                            content = match.group(1) if match.lastindex >= 1 else ''
                            content = re.sub(r'\s+', ' ', content.strip())
                            content = re.sub(r',\s*', ', ', content)
                            return f'"operands": [{content}]'
                        operands_pattern = r'"operands":\s*\[\s*((?:"[^"]*"(?:\s*,\s*"[^"]*")*)?)\s*\]'
                        json_str = re.sub(operands_pattern, format_simple_operands, json_str, flags=re.DOTALL)
                        
                        # 保存到新文件
                        with open(new_file_path, 'w', encoding='utf-8') as f:
                            f.write(json_str)
                        
                        # 更新session_state中的文件路径
                        st.session_state.loaded_json_file_path = new_file_path
                        st.session_state.last_uploaded_json_name = new_filename
                        
                        st.success(f"✅ 文件已保存: `{new_filename}`（原文件已重命名为 `{backup_filename}`）")
                        
                        # 下载按钮
                        with open(new_file_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载JSON文件",
                                data=f.read(),
                                file_name=os.path.basename(new_file_path),
                                mime="application/json"
                            )
                    except Exception as e:
                        st.error(f"❌ 保存文件失败: {e}")
                else:
                    # 普通保存（新文件）
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    output_dir = os.path.join(base_dir, 'json_save')
                    
                    # 注意：model_name已经包含在每个feature对象中，不需要单独传递
                    filepath = save_features_to_json(
                        st.session_state.generated_features,
                        target_name,
                        feature_names,
                        output_dir
                    )
                    
                    if filepath:
                        st.success(f"✅ 文件已保存: `{filepath}`")
                        
                        # 下载按钮
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                label="📥 下载JSON文件",
                                data=f.read(),
                                file_name=os.path.basename(filepath),
                                mime="application/json"
                            )
                    else:
                        st.error("❌ 保存文件失败")
            else:
                st.warning("⚠️ 没有可保存的特征")

