"""
Streamlit前端应用
用于LLM特征生成
"""
import streamlit as st
import sys
import os
import logging
import time
from datetime import datetime, timedelta

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加父目录到路径（用于导入llm_new_feature模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_new_feature.feature_generator import generate_features

# 确保当前目录在sys.path的最前面，并清理可能存在的utils模块缓存
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
else:
    # 如果已经在sys.path中，移到最前面
    sys.path.remove(current_dir)
    sys.path.insert(0, current_dir)

# 清理可能存在的模块缓存（避免导入错误的模块）
modules_to_clear = ['utils', 'llm_new_feature.llm_api', 'llm_new_feature.llm_prompt', 'llm_new_feature']
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]

# 导入当前目录下的utils模块
from utils import (
    init_session_state,
    render_sidebar,
    render_feature_config,
    render_prompt_config,
    render_api_config,
    render_feature_display
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LLM特征生成工具",
    page_icon="🧬",
    layout="wide"
)


def execute_feature_generation(
    target_column,
    feature_columns,
    num_features,
    task_context,
    api_config,
    show_success=True,
    max_height=None
):
    """
    执行特征生成的通用函数
    
    Returns:
        bool: 生成是否成功
    """
    if len(feature_columns) == 0:
        if show_success:
            st.error("请至少保留一个特征列（不要删除所有列）")
        return False
    
    try:
        # 获取已存在的特征tree列表（用于避免重复）
        existing_trees_for_generation = []
        if st.session_state.generated_features:
            existing_trees_for_generation = [
                feature.get('tree') 
                for feature in st.session_state.generated_features 
                if feature.get('tree')
            ]
        
        # 生成特征（不自动保存），同时获取原始响应
        result = generate_features(
            target_name=target_column,
            feature_names=feature_columns,
            num_features=num_features,
            task_context=task_context if task_context.strip() else None,
            output_dir=None,
            existing_trees=existing_trees_for_generation if existing_trees_for_generation else None,
            api_config=api_config,
            return_raw_response=True,  # 返回原始响应
            max_height=max_height
        )
        
        # 处理返回值（可能是字典或列表，取决于return_raw_response参数）
        if isinstance(result, dict) and 'features' in result:
            generated_features = result['features']
            raw_responses = result.get('raw_responses', [])
        else:
            # 向后兼容：如果不是字典，说明是旧的返回值格式
            generated_features = result
            raw_responses = []
        
        # 保存原始响应到session state（无论成功或失败都要保存）
            if raw_responses:
                if 'llm_raw_responses' not in st.session_state:
                    st.session_state.llm_raw_responses = []
                # 追加新的原始响应
                for raw_resp in raw_responses:
                    st.session_state.llm_raw_responses.append({
                        'target_name': target_column,
                        'feature_names': feature_columns,
                        'num_features': num_features,
                        'attempt': raw_resp.get('attempt', 1),
                        'response': raw_resp.get('response', ''),
                    'timestamp': raw_resp.get('timestamp', time.time()),
                    'error': raw_resp.get('error', False)  # 保存错误标记
                    })
        
        if generated_features and len(generated_features) > 0:
            # 为每个特征添加模型名称
            model_name = api_config.get('model', 'Unknown')
            for feature in generated_features:
                feature['model_name'] = model_name
            
            if show_success:
                st.success(f"✅ 成功生成 {len(generated_features)} 个特征！")
            # 累计追加到session state（而不是替换）
            if st.session_state.generated_features:
                st.session_state.generated_features.extend(generated_features)
                if show_success:
                    st.success(f"📊 当前共有 {len(st.session_state.generated_features)} 个特征（累计）")
            else:
                st.session_state.generated_features = generated_features
            st.session_state.target_name = target_column
            st.session_state.feature_names = feature_columns
            st.session_state.model_name = model_name
            # 生成特征后，自动更新 prompt（增加计数器以触发更新）
            st.session_state.prompt_update_counter += 1
            # 清除用户编辑的 prompt，以便使用新的 prompt（包含新生成的特征信息）
            st.session_state.user_edited_prompt = None
            return True
        else:
            if show_success:
                st.error("❌ 特征生成失败，请查看日志")
            return False
    except Exception as e:
        if show_success:
            st.error(f"生成特征时出错: {e}")
        logger.exception("生成特征时出错")
        return False


def main():
    st.title("🧬 LLM特征生成工具")
    st.markdown("---")
    
    # 初始化session state
    init_session_state()
    
    # 渲染侧边栏
    df = render_sidebar()
    
    # 处理追加特征触发（在显示主界面之前）
    if st.session_state.get('append_features_triggered') and not st.session_state.get('append_config_set'):
        if st.session_state.target_name and st.session_state.feature_names:
            # 根据JSON文件自动设置配置（不需要数据集）
            feature_cols_from_json = st.session_state.feature_names
            
            # 如果数据集存在，设置目标变量和排除列；如果不存在，只需要设置基本信息
            if df is not None and st.session_state.target_name in df.columns:
                # 设置目标变量
                st.session_state.target_column_selectbox = st.session_state.target_name
                
                # 设置排除列（排除目标变量和不在特征列表中的列）
                all_cols = df.columns.tolist()
                target_col = st.session_state.target_name
                exclude_cols = [col for col in all_cols if col != target_col and col not in feature_cols_from_json]
                st.session_state.exclude_columns_multiselect = exclude_cols
            
            # 设置模型名称（如果JSON中有）
            if st.session_state.generated_features and st.session_state.generated_features[0].get('model_name'):
                st.session_state.model_name = st.session_state.generated_features[0].get('model_name')
            
            st.session_state.append_config_set = True
            st.session_state.append_features_triggered = False  # 重置触发标志
            st.success(f"✅ 已根据JSON文件自动配置：目标变量={st.session_state.target_name}，特征列={len(feature_cols_from_json)}个")
            if df is None:
                st.info("💡 如需生成新特征，请先上传数据集")
            st.rerun()
        else:
            st.error("❌ 无法获取JSON文件的配置信息，请重新加载JSON文件")
            st.session_state.append_features_triggered = False
    
    # 主界面
    if df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("数据集预览")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("数据集信息")
            st.metric("样本数", df.shape[0])
            st.metric("特征数", df.shape[1])
        
        st.markdown("---")
        
        # 特征配置
        target_column, exclude_columns, num_features, feature_columns, max_height = render_feature_config(df)
        
        st.markdown("---")
        
        # Prompt配置
        task_context, avoid_duplicates = render_prompt_config(
            target_column, exclude_columns, feature_columns, num_features, max_height
        )
        
        st.markdown("---")
        
        # API配置
        api_config = render_api_config()
        
        st.markdown("---")
        
        # 显示LLM原始响应（移到生成特征之前）
        if st.session_state.llm_raw_responses and len(st.session_state.llm_raw_responses) > 0:
            st.subheader("5. LLM原始响应")
            
            # 显示/隐藏切换按钮
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📝 共保存了 {len(st.session_state.llm_raw_responses)} 条LLM原始响应记录")
            with col2:
                if st.button("🗑️ 清空原始响应", type="secondary", key="clear_responses_top"):
                    st.session_state.llm_raw_responses = []
                    st.rerun()
            
            # 响应选择器（显示最近的几条）
            # 当只有1条响应时，直接显示，不使用slider
            total_responses = len(st.session_state.llm_raw_responses)
            if total_responses == 1:
                num_to_show = 1
            else:
                num_to_show = st.slider(
                    "选择要查看的响应数量",
                    min_value=1,
                    max_value=min(10, total_responses),
                    value=min(3, total_responses),
                    help="选择要显示的最近几条原始响应",
                    key="response_slider_top"
                )
            
            # 显示选中的响应（从最近的开始）
            recent_responses = st.session_state.llm_raw_responses[-num_to_show:]
            recent_responses.reverse()  # 最新的在前
            
            for idx, raw_resp in enumerate(recent_responses):
                # 计算响应编号（从1开始，最新的编号最大）
                response_idx = len(st.session_state.llm_raw_responses) - num_to_show + idx + 1
                
                # 格式化时间戳
                timestamp = raw_resp.get('timestamp')
                if timestamp:
                    try:
                        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        time_str = 'Unknown'
                else:
                    time_str = 'Unknown'
                
                # 检查是否有错误
                is_error = raw_resp.get('error', False)
                error_marker = "❌ " if is_error else ""
                
                with st.expander(
                    f"{error_marker}响应 #{response_idx} - "
                    f"目标: {raw_resp.get('target_name', 'Unknown')} | "
                    f"生成数量: {raw_resp.get('num_features', 0)} | "
                    f"尝试次数: {raw_resp.get('attempt', 1)} | "
                    f"时间: {time_str}",
                    expanded=(idx == 0)  # 默认展开第一条
                ):
                    # 显示响应信息
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.text(f"目标变量: {raw_resp.get('target_name', 'Unknown')}")
                        st.text(f"特征列数: {len(raw_resp.get('feature_names', []))}")
                    with col_info2:
                        st.text(f"请求生成数: {raw_resp.get('num_features', 0)}")
                        st.text(f"尝试次数: {raw_resp.get('attempt', 1)}")
                        if is_error:
                            st.error("⚠️ API调用失败")
                    
                    # 显示原始响应内容
                    raw_response_text = raw_resp.get('response', '')
                    st.markdown("**原始响应内容：**")
                    st.code(raw_response_text, language='text', line_numbers=True)
                    
                    # 显示响应长度统计
                    response_length = len(raw_response_text)
                    st.caption(f"响应长度: {response_length} 字符")
            
            if len(st.session_state.llm_raw_responses) > num_to_show:
                st.info(f"💡 只显示最近 {num_to_show} 条响应，共有 {len(st.session_state.llm_raw_responses)} 条记录")
        
        st.markdown("---")
        
        # 生成按钮
        st.subheader("6. 生成特征")
        
        # 处理自动重复触发的逻辑
        should_generate = False
        is_auto_repeat = False
        
        # 检查是否正在等待下一次生成
        if st.session_state.auto_repeat_waiting:
            wait_until = st.session_state.get('auto_repeat_wait_until', None)
            if wait_until:
                if datetime.now() >= wait_until:
                    # 等待时间已到，触发下一次生成
                    st.session_state.auto_repeat_waiting = False
                    st.session_state.auto_repeat_wait_until = None
                    st.session_state.auto_repeat_trigger = True
                else:
                    # 还在等待中，显示等待信息
                    remaining = (wait_until - datetime.now()).total_seconds()
                    if remaining > 0.5:
                        st.info(f"⏳ 等待中，将在 {remaining:.1f} 秒后继续下一次生成...")
                        time.sleep(0.5)  # 短暂等待后刷新
                        st.rerun()
                    else:
                        # 时间快到了，直接触发
                        st.session_state.auto_repeat_waiting = False
                        st.session_state.auto_repeat_wait_until = None
                        st.session_state.auto_repeat_trigger = True
        
        # 检查是否应该自动重复生成（通过 trigger 触发）
        if st.session_state.auto_repeat_trigger and st.session_state.auto_repeat_mode:
            should_generate = True
            is_auto_repeat = True
            st.session_state.auto_repeat_trigger = False
        
        # 自动重复次数选择（在按钮上方）
        col_repeat = st.columns([3, 2, 3])
        with col_repeat[1]:
            repeat_times = st.number_input(
                "🔄 自动重复次数",
                min_value=1,
                max_value=100,
                value=st.session_state.auto_repeat_total_times,
                step=1,
                help="设置自动重复生成特征的次数（每次间隔2秒）",
                disabled=st.session_state.auto_repeat_mode  # 正在运行时禁用修改
            )
            if not st.session_state.auto_repeat_mode:
                st.session_state.auto_repeat_total_times = repeat_times
        
        # 生成按钮行
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 开始生成特征", type="primary", use_container_width=True):
                should_generate = True
                is_auto_repeat = False
                # 如果是手动点击，停止自动重复模式
                st.session_state.auto_repeat_mode = False
                st.session_state.auto_repeat_trigger = False
                st.session_state.auto_repeat_waiting = False
        
        with col2:
            # 自动重复生成按钮
            auto_repeat_label = "⏸️ 停止自动重复" if st.session_state.auto_repeat_mode else "🔄 自动重复生成"
            auto_repeat_type = "secondary" if st.session_state.auto_repeat_mode else "secondary"
            if st.button(auto_repeat_label, type=auto_repeat_type, use_container_width=True):
                if st.session_state.auto_repeat_mode:
                    # 停止自动重复模式
                    st.session_state.auto_repeat_mode = False
                    st.session_state.auto_repeat_trigger = False
                    st.session_state.auto_repeat_waiting = False
                    st.session_state.auto_repeat_count = 0  # 重置计数器
                    st.success("已停止自动重复生成")
                    st.rerun()
                else:
                    # 开始自动重复模式
                    if repeat_times < 1:
                        st.error("自动重复次数必须至少为1")
                    else:
                        st.session_state.auto_repeat_mode = True
                        st.session_state.auto_repeat_trigger = True
                        st.session_state.auto_repeat_waiting = False
                        st.session_state.auto_repeat_count = 0  # 重置计数器
                        st.session_state.auto_repeat_total_times = repeat_times  # 保存用户选择
                        should_generate = True
                        is_auto_repeat = True
                        st.info(f"🔄 已启动自动重复生成模式，将自动执行 {repeat_times} 次生成（每次间隔2秒）")
        
        # 显示自动重复状态
        if st.session_state.auto_repeat_mode:
            total_times = st.session_state.auto_repeat_total_times
            current_count = st.session_state.auto_repeat_count
            remaining = total_times - current_count
            st.info(f"🔄 自动重复生成模式已激活 - 已完成 {current_count}/{total_times} 次，剩余 {remaining} 次")
        
        # 执行特征生成
        if should_generate:
            spinner_text = "正在自动重复生成特征，请稍候..." if is_auto_repeat else "正在生成特征，请稍候..."
            with st.spinner(spinner_text):
                success = execute_feature_generation(
                    target_column=target_column,
                    feature_columns=feature_columns,
                    num_features=num_features,
                    task_context=task_context,
                    api_config=api_config,
                    show_success=True,
                    max_height=max_height
                )
                
                if success:
                    # 如果生成成功且处于自动重复模式，继续执行后续生成
                    if st.session_state.auto_repeat_mode:
                        # 初始化计数器（如果还没有）
                        if 'auto_repeat_count' not in st.session_state:
                            st.session_state.auto_repeat_count = 0
                        
                        # 增加计数器（表示已完成一次生成）
                        st.session_state.auto_repeat_count += 1
                        
                        total_times = st.session_state.auto_repeat_total_times
                        current_count = st.session_state.auto_repeat_count
                        
                        # 检查是否还需要继续生成
                        if current_count < total_times:
                            # 还有剩余次数，设置等待状态，然后刷新页面以更新 prompt
                            remaining = total_times - current_count
                            total_features_count = len(st.session_state.generated_features) if st.session_state.generated_features else 0
                            st.success(f"✅ 第 {current_count} 次生成完成！当前共有 {total_features_count} 个特征。")
                            st.info(f"📝 Prompt 已自动更新（包含 {total_features_count} 个已存在的特征）。等待 2 秒后自动进行第 {current_count + 1} 次生成（剩余 {remaining} 次）...")
                            
                            # 设置等待状态（等待2秒）
                            st.session_state.auto_repeat_waiting = True
                            st.session_state.auto_repeat_wait_until = datetime.now() + timedelta(seconds=2)
                            
                            # 刷新页面，让 prompt 更新显示
                            st.rerun()
                        else:
                            # 已经完成所有生成
                            st.success(f"🎉 自动重复生成完成！已成功生成 {total_times} 次特征。")
                            # 重置状态
                            st.session_state.auto_repeat_mode = False
                            st.session_state.auto_repeat_trigger = False
                            st.session_state.auto_repeat_waiting = False
                            st.session_state.auto_repeat_count = 0
                            st.session_state.auto_repeat_wait_until = None
                            st.rerun()
                    else:
                        # 非自动重复模式，正常刷新
                        st.rerun()
                else:
                    # 生成失败，停止自动重复模式
                    if st.session_state.auto_repeat_mode:
                        st.session_state.auto_repeat_mode = False
                        st.session_state.auto_repeat_trigger = False
                        st.session_state.auto_repeat_waiting = False
                        st.session_state.auto_repeat_count = 0
                        st.session_state.auto_repeat_wait_until = None
        
        st.markdown("---")
    
    # 如果没有数据集，但已加载JSON并点击了追加特征，显示配置信息
    elif df is None and st.session_state.get('append_config_set') and st.session_state.target_name and st.session_state.feature_names:
        st.info("📋 追加特征模式：当前使用JSON文件中的配置，无需数据集")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("当前配置（来自JSON文件）")
            st.info(f"**目标变量**: {st.session_state.target_name}")
            st.info(f"**特征列**: {len(st.session_state.feature_names)} 个特征")
            if len(st.session_state.feature_names) <= 20:
                st.text(f"特征列表: {', '.join(st.session_state.feature_names)}")
            else:
                st.text(f"特征列表: {', '.join(st.session_state.feature_names[:20])} ... 等 {len(st.session_state.feature_names)} 个特征")
        
        with col2:
            st.subheader("配置信息")
            num_features = st.number_input(
                "生成特征数量",
                min_value=1,
                max_value=1000,
                value=10,
                help="要生成的LLM特征数量",
                key="num_features_input_no_df"
            )
            max_height = st.number_input(
                "表达式高度限制",
                min_value=0,
                max_value=10,
                value=1,
                help="生成的表达式树的高度限制（采用DEAP标准）。高度计算规则：叶子节点（特征名）height为0，内部节点：height = 1 + max(所有操作数的height)。例如：A+B height为1，max(A+B, C/D) height为2。",
                key="max_height_input_no_df"
            )
        
        st.markdown("---")
        
        # Prompt配置（使用JSON中的信息）
        task_context, avoid_duplicates = render_prompt_config(
            st.session_state.target_name,
            [],  # 没有排除列
            st.session_state.feature_names,  # 使用JSON中的特征列
            num_features,
            max_height
        )
        
        st.markdown("---")
        
        # API配置
        api_config = render_api_config()
        
        st.markdown("---")
        
        # 生成按钮
        st.subheader("6. 生成特征")
        
        # 处理自动重复触发的逻辑
        should_generate = False
        is_auto_repeat = False
        
        # 检查是否正在等待下一次生成
        if st.session_state.auto_repeat_waiting:
            wait_until = st.session_state.get('auto_repeat_wait_until', None)
            if wait_until:
                if datetime.now() >= wait_until:
                    # 等待时间已到，触发下一次生成
                    st.session_state.auto_repeat_waiting = False
                    st.session_state.auto_repeat_wait_until = None
                    st.session_state.auto_repeat_trigger = True
                else:
                    # 还在等待中，显示等待信息
                    remaining = (wait_until - datetime.now()).total_seconds()
                    if remaining > 0.5:
                        st.info(f"⏳ 等待中，将在 {remaining:.1f} 秒后继续下一次生成...")
                        time.sleep(0.5)  # 短暂等待后刷新
                        st.rerun()
                    else:
                        # 时间快到了，直接触发
                        st.session_state.auto_repeat_waiting = False
                        st.session_state.auto_repeat_wait_until = None
                        st.session_state.auto_repeat_trigger = True
        
        # 检查是否应该自动重复生成（通过 trigger 触发）
        if st.session_state.auto_repeat_trigger and st.session_state.auto_repeat_mode:
            should_generate = True
            is_auto_repeat = True
            st.session_state.auto_repeat_trigger = False
        
        # 自动重复次数选择（在按钮上方）
        col_repeat = st.columns([3, 2, 3])
        with col_repeat[1]:
            repeat_times = st.number_input(
                "🔄 自动重复次数",
                min_value=1,
                max_value=100,
                value=st.session_state.auto_repeat_total_times,
                step=1,
                help="设置自动重复生成特征的次数（每次间隔2秒）",
                disabled=st.session_state.auto_repeat_mode,  # 正在运行时禁用修改
                key="repeat_times_no_df"
            )
            if not st.session_state.auto_repeat_mode:
                st.session_state.auto_repeat_total_times = repeat_times
        
        # 生成按钮行
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 开始生成特征", type="primary", use_container_width=True, key="generate_features_no_df"):
                should_generate = True
                is_auto_repeat = False
                # 如果是手动点击，停止自动重复模式
                st.session_state.auto_repeat_mode = False
                st.session_state.auto_repeat_trigger = False
                st.session_state.auto_repeat_waiting = False
        
        with col2:
            # 自动重复生成按钮
            auto_repeat_label = "⏸️ 停止自动重复" if st.session_state.auto_repeat_mode else "🔄 自动重复生成"
            auto_repeat_type = "secondary" if st.session_state.auto_repeat_mode else "secondary"
            if st.button(auto_repeat_label, type=auto_repeat_type, use_container_width=True, key="auto_repeat_no_df"):
                if st.session_state.auto_repeat_mode:
                    # 停止自动重复模式
                    st.session_state.auto_repeat_mode = False
                    st.session_state.auto_repeat_trigger = False
                    st.session_state.auto_repeat_waiting = False
                    st.session_state.auto_repeat_count = 0  # 重置计数器
                    st.success("已停止自动重复生成")
                    st.rerun()
                else:
                    # 开始自动重复模式
                    if repeat_times < 1:
                        st.error("自动重复次数必须至少为1")
                    else:
                        st.session_state.auto_repeat_mode = True
                        st.session_state.auto_repeat_trigger = True
                        st.session_state.auto_repeat_waiting = False
                        st.session_state.auto_repeat_count = 0  # 重置计数器
                        st.session_state.auto_repeat_total_times = repeat_times  # 保存用户选择
                        should_generate = True
                        is_auto_repeat = True
                        st.info(f"🔄 已启动自动重复生成模式，将自动执行 {repeat_times} 次生成（每次间隔2秒）")
        
        # 显示自动重复状态
        if st.session_state.auto_repeat_mode:
            total_times = st.session_state.auto_repeat_total_times
            current_count = st.session_state.auto_repeat_count
            remaining = total_times - current_count
            st.info(f"🔄 自动重复生成模式已激活 - 已完成 {current_count}/{total_times} 次，剩余 {remaining} 次")
        
        # 执行特征生成
        if should_generate:
            spinner_text = "正在自动重复生成特征，请稍候..." if is_auto_repeat else "正在生成特征，请稍候..."
            with st.spinner(spinner_text):
                success = execute_feature_generation(
                    target_column=st.session_state.target_name,
                    feature_columns=st.session_state.feature_names,
                    num_features=num_features,
                    task_context=task_context,
                    api_config=api_config,
                    show_success=True,
                    max_height=max_height
                )
                
                if success:
                    # 如果生成成功且处于自动重复模式，继续执行后续生成
                    if st.session_state.auto_repeat_mode:
                        # 初始化计数器（如果还没有）
                        if 'auto_repeat_count' not in st.session_state:
                            st.session_state.auto_repeat_count = 0
                        
                        # 增加计数器（表示已完成一次生成）
                        st.session_state.auto_repeat_count += 1
                        
                        total_times = st.session_state.auto_repeat_total_times
                        current_count = st.session_state.auto_repeat_count
                        
                        # 检查是否还需要继续生成
                        if current_count < total_times:
                            # 还有剩余次数，设置等待状态，然后刷新页面以更新 prompt
                            remaining = total_times - current_count
                            total_features_count = len(st.session_state.generated_features) if st.session_state.generated_features else 0
                            st.success(f"✅ 第 {current_count} 次生成完成！当前共有 {total_features_count} 个特征。")
                            st.info(f"📝 Prompt 已自动更新（包含 {total_features_count} 个已存在的特征）。等待 2 秒后自动进行第 {current_count + 1} 次生成（剩余 {remaining} 次）...")
                            
                            # 设置等待状态（等待2秒）
                            st.session_state.auto_repeat_waiting = True
                            st.session_state.auto_repeat_wait_until = datetime.now() + timedelta(seconds=2)
                            
                            # 刷新页面，让 prompt 更新显示
                            st.rerun()
                        else:
                            # 已经完成所有生成
                            st.success(f"🎉 自动重复生成完成！已成功生成 {total_times} 次特征。")
                            # 重置状态
                            st.session_state.auto_repeat_mode = False
                            st.session_state.auto_repeat_trigger = False
                            st.session_state.auto_repeat_waiting = False
                            st.session_state.auto_repeat_count = 0
                            st.session_state.auto_repeat_wait_until = None
                            st.rerun()
                    else:
                        # 非自动重复模式，正常刷新
                        st.rerun()
                else:
                    # 生成失败，停止自动重复模式
                    if st.session_state.auto_repeat_mode:
                        st.session_state.auto_repeat_mode = False
                        st.session_state.auto_repeat_trigger = False
                        st.session_state.auto_repeat_waiting = False
                        st.session_state.auto_repeat_count = 0
                        st.session_state.auto_repeat_wait_until = None
    
    st.markdown("---")
    
    # 显示生成的特征（表格形式）- 无论是否有数据集都可以显示
    render_feature_display()
    
    # 如果没有数据集，但也没有特征，显示提示信息
    if df is None and not st.session_state.generated_features and not st.session_state.get('append_config_set'):
        st.info("👈 请先在侧边栏上传数据集文件或JSON特征文件")
        
        # 显示示例
        st.markdown("---")
        st.subheader("使用说明")
        st.markdown("""
        1. **上传数据集**: 在侧边栏上传包含特征列的CSV文件
        2. **选择目标变量**: 选择要预测的目标列
        3. **删除列**: 选择要排除的列，其余列将作为特征列
        4. **设置生成数量**: 输入要生成的特征数量（1-1000）
        5. **编辑Prompt**: 查看和编辑完整的LLM prompt
        6. **配置API**: 设置API密钥、URL、模型等参数
        7. **生成特征**: 点击按钮开始生成
        8. **编辑特征**: 在表格中删除不需要的特征
        9. **保存文件**: 点击保存按钮保存为JSON文件
        """)
        
        st.markdown("---")
        st.subheader("输出格式")
        st.code("""
{
  "target_name": "Ash_Deformation",
  "feature_names": ["SiO2", "Al2O3", ...],
  "num_features": 10,
  "generated_at": "20251128_204856",
  "features": [
    {
      "tree": {
        "operator": "Div",
        "operands": ["SiO2", "Al2O3"]
      },
      "description": "硅铝比，反映灰分中主要酸性氧化物的比例关系",
      "notation": "SiO₂ / Al₂O₃"
    },
    ...
  ]
}
        """, language="json")


if __name__ == "__main__":
    main()
