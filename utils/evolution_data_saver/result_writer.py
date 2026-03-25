"""
结果写入模块
负责将进化结果写入文本文件
"""

import os
import json as _json_reader
from datetime import datetime
from typing import List, Optional


def _get_dataset_dir_from_csv(csv_path: str) -> str:
    """返回CSV所在的数据集目录（其下含有result目录）"""
    return os.path.dirname(csv_path)


def _ensure_result_dirs(dataset_dir: str) -> str:
    """确保 result 目录存在并返回其路径。"""
    result_dir = os.path.join(dataset_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def write_evo_result(
    csv_path: str,
    feature_names: List[str],
    target_name: str,
    dropped_columns: Optional[List[str]],
    seed: int,
    train_metrics: dict,
    test_metrics: dict,
    expression: str,
    run_time: Optional[str] = None,
    evolution_json_path: Optional[str] = None,
    ridge_formula: Optional[str] = None,
    baseline_results: Optional[List[dict]] = None,
    cv_metrics: Optional[dict] = None,
    output_dir: Optional[str] = None,  # 新增：指定输出目录，如果提供则使用此目录而不是基于CSV路径
    residual_fitting_results: Optional[dict] = None,  # 新增：残差拟合结果
    training_duration: Optional[float] = None,  # 新增：核心训练时间（秒）
    residual_training_duration: Optional[float] = None,  # 新增：残差模型训练时间（秒）
    population_init_info: Optional[dict] = None,  # 新增：种群初始化统计信息
    dynamic_expansion_logs: Optional[List[str]] = None,  # 新增：动态分支扩展日志
    train_file_path: Optional[str] = None,  # 新增：训练集路径
    test_file_path: Optional[str] = None,  # 新增：测试集路径
):
    """
    写入/追加 evo_result.txt，精确保留一行本次 run 的结果。
    
    如果提供了output_dir，则在output_dir下创建文件。
    否则，在与CSV同级的 result 目录下创建文件（兼容旧行为）。

    格式示例（参考提供的格式）：
      CSV: <file>
      Features used: [f1, f2, ...]
      Target: <target>
      Dropped columns: [..]
      Random Seed: <seed>
      Train: R2: ..., MSE: ..., RMSE: ..., MAE: ...
      Test:  R2: ..., MSE: ..., RMSE: ..., MAE: ...
      Expression: <best individual string>
      Run time: yyyy-mm-dd HH:MM:SS
      Evolution JSON: <path>
      [进化过程统计表格]
      ================================================================================
    """
    if output_dir:
        # 如果提供了output_dir，直接使用该目录
        os.makedirs(output_dir, exist_ok=True)
        evo_file = os.path.join(output_dir, "evo_result.txt")
    else:
        # 兼容旧行为：基于CSV路径创建result目录
        dataset_dir = _get_dataset_dir_from_csv(csv_path)
        result_dir = _ensure_result_dirs(dataset_dir)
        # 将统计文件写到目标子目录下
        if target_name:
            target_dir = os.path.join(result_dir, target_name)
            os.makedirs(target_dir, exist_ok=True)
            evo_file = os.path.join(target_dir, "evo_result.txt")
        else:
            evo_file = os.path.join(result_dir, "evo_result.txt")

    timestamp = run_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_str = '[' + ', '.join(feature_names) + ']'

    # 尝试从JSON文件读取train_file_path和test_file_path（如果未提供）
    json_train_path = train_file_path
    json_test_path = test_file_path
    json_baseline_info = None
    json_gp_training_duration = None
    
    if evolution_json_path and os.path.exists(evolution_json_path):
        try:
            with open(evolution_json_path, 'r', encoding='utf-8') as jf:
                evo_json = _json_reader.load(jf)
            
            # 从experiment_info读取路径
            if 'experiment_info' in evo_json:
                exp_info = evo_json['experiment_info']
                if json_train_path is None and 'train_file_path' in exp_info:
                    json_train_path = exp_info['train_file_path']
                if json_test_path is None and 'test_file_path' in exp_info:
                    json_test_path = exp_info['test_file_path']
            
            # 从baseline_info读取baseline结果
            if 'baseline_info' in evo_json:
                json_baseline_info = evo_json['baseline_info']
            
            # 从gp_info读取训练时间
            if 'gp_info' in evo_json and 'gp_final_model' in evo_json['gp_info']:
                final_model = evo_json['gp_info']['gp_final_model']
                if 'training_time' in final_model and 'duration_seconds' in final_model['training_time']:
                    json_gp_training_duration = final_model['training_time']['duration_seconds']
        except Exception as e:
            pass  # 如果读取失败，使用默认值

    lines = []
    # Experiment_Info 部分
    lines.append("Experiment_Info:")
    lines.append("")
    if json_train_path:
        lines.append(f"Train Set: {json_train_path}")
    else:
        lines.append(f"Train Set: {os.path.basename(csv_path)}")
    if json_test_path:
        lines.append(f"Test Set: {json_test_path}")
    else:
        lines.append(f"Test Set: {os.path.basename(csv_path)}")
    lines.append(f"Features used: {features_str}")
    lines.append(f"Target: {target_name}")
    lines.append(f"Random Seed: {seed}")
    lines.append("")
    
    # Baseline_Info 部分
    lines.append("Baseline_Info:")
    lines.append("")
    # 从JSON或参数中获取baseline信息
    baseline_models = None
    if json_baseline_info and 'baseline_models' in json_baseline_info:
        baseline_models = json_baseline_info['baseline_models']
    elif baseline_results:
        baseline_models = baseline_results
    
    if baseline_models:
        # 获取残差拟合结果（如果有），构建模型名称到测试集R²的映射
        residual_test_r2_map = {}  # {model_name: test_r2}
        if residual_fitting_results and isinstance(residual_fitting_results, dict):
            with_residual = residual_fitting_results.get('with_residual', {})
            # 支持新格式（列表）和旧格式（字典）
            if isinstance(with_residual, list) and len(with_residual) > 0:
                # 新格式：列表，遍历所有残差模型
                for residual_item in with_residual:
                    model_name = residual_item.get('model_name', 'Unknown')
                    residual_test = residual_item.get('test', {})
                    residual_test_r2 = residual_test.get('r2')
                    if residual_test_r2 is not None:
                        residual_test_r2_map[model_name] = residual_test_r2
            elif isinstance(with_residual, dict):
                # 旧格式：字典
                residual_model_name = "RandomForest"  # 默认值
                residual_model = with_residual.get('residual_model')
                if residual_model is not None:
                    try:
                        class_name = residual_model.__class__.__name__
                        if 'RandomForest' in class_name:
                            residual_model_name = "RandomForest"
                        elif 'Ridge' in class_name:
                            residual_model_name = "RidgeCV"
                        elif 'CatBoost' in class_name:
                            residual_model_name = "CatBoost"
                        elif 'ExtraTrees' in class_name:
                            residual_model_name = "ExtraTrees"
                        elif 'GradientBoosting' in class_name:
                            residual_model_name = "GradientBoosting"
                        elif 'LightGBM' in class_name:
                            residual_model_name = "LightGBM"
                    except:
                        pass
                residual_test = with_residual.get('test', {})
                residual_test_r2 = residual_test.get('r2')
                if residual_test_r2 is not None:
                    residual_test_r2_map[residual_model_name] = residual_test_r2
        
        # 收集所有列的数据
        column_data = {
            'model_name': [],
            'cv_r2': [],
            'test_r2': [],
            'test_r2_with_residual': [],
            'train_r2': [],
            'training_time': []
        }
        
        # 遍历所有baseline模型，收集数据
        for model in baseline_models:
            model_name = model.get('model_name', 'Unknown')
            
            # 获取指标
            cv_r2_mean = None
            cv_r2_std = None
            test_r2 = None
            train_r2 = None
            training_time = None
            
            if 'cv_metrics' in model and 'cross_validation' in model['cv_metrics']:
                cross_validation = model['cv_metrics']['cross_validation']
                cv_r2_mean = cross_validation.get('r2_mean', 0)
                cv_r2_std = cross_validation.get('r2_std', 0)
            if 'test_metrics' in model:
                test_r2 = model['test_metrics'].get('r2', 0)
            if 'train_metrics' in model:
                train_r2 = model['train_metrics'].get('r2', 0)
            if 'training_duration' in model:
                if isinstance(model['training_duration'], dict):
                    training_time = model['training_duration'].get('seconds', 0)
                else:
                    training_time = model['training_duration']
            
            # 格式化输出（CV_R² 显示为均值±标准差）
            if cv_r2_mean is not None and cv_r2_std is not None:
                cv_r2_str = f"{cv_r2_mean:.6f}±{cv_r2_std:.6f}"
            elif cv_r2_mean is not None:
                cv_r2_str = f"{cv_r2_mean:.6f}"
            else:
                cv_r2_str = "N/A"
            
            test_r2_str = f"{test_r2:.6f}" if test_r2 is not None else "N/A"
            train_r2_str = f"{train_r2:.6f}" if train_r2 is not None else "N/A"
            training_time_str = f"{training_time:.6f}" if training_time is not None else "N/A"
            
            # 残差拟合的Test_R²：根据模型名称匹配对应的残差拟合结果
            # 1. 优先检查模型数据中是否包含残差拟合结果
            model_residual_test_r2 = None
            if 'residual_fitting' in model or 'test_r2_with_residual' in model:
                # 如果模型数据中包含残差拟合结果，则使用它
                model_residual_test_r2 = model.get('test_r2_with_residual') or \
                    (model.get('residual_fitting', {}).get('test', {}).get('r2') if isinstance(model.get('residual_fitting'), dict) else None)
            
            # 2. 如果模型自身没有残差拟合结果，则从GP残差拟合结果中查找匹配的模型
            if model_residual_test_r2 is None:
                # 尝试匹配模型名称（支持多种名称变体）
                matched_r2 = None
                # 直接匹配
                if model_name in residual_test_r2_map:
                    matched_r2 = residual_test_r2_map[model_name]
                else:
                    # 尝试部分匹配（处理名称变体）
                    for residual_model_name, test_r2 in residual_test_r2_map.items():
                        if model_name.lower() == residual_model_name.lower() or \
                           model_name.lower() in residual_model_name.lower() or \
                           residual_model_name.lower() in model_name.lower():
                            matched_r2 = test_r2
                            break
                
                model_residual_test_r2 = matched_r2
            
            # 格式化输出
            if model_residual_test_r2 is not None:
                test_r2_with_residual_str = f"{model_residual_test_r2:.6f}"
            else:
                test_r2_with_residual_str = "null"
            
            column_data['model_name'].append(model_name)
            column_data['cv_r2'].append(cv_r2_str)
            column_data['test_r2'].append(test_r2_str)
            column_data['test_r2_with_residual'].append(test_r2_with_residual_str)
            column_data['train_r2'].append(train_r2_str)
            column_data['training_time'].append(training_time_str)
        
        # 计算每列的最大长度（包括标题）
        headers = ['', 'CV_R²', 'Test_R²', 'Test_R²_with_residual', 'Train_R²', 'training_time']
        column_keys = ['model_name', 'cv_r2', 'test_r2', 'test_r2_with_residual', 'train_r2', 'training_time']
        
        max_lengths = []
        for i, key in enumerate(column_keys):
            # 第一列没有标题，只从数据中计算最大长度
            max_len = len(headers[i]) if i > 0 and headers[i] else 0
            if key in column_data:
                for value in column_data[key]:
                    max_len = max(max_len, len(str(value)))
            max_lengths.append(max_len)
        
        # 计算每列需要的制表符数量（根据最大长度）
        tab_width = 8  # 制表符通常为8个字符宽度
        tabs_per_column = []
        for max_len in max_lengths:
            # 计算需要多少个制表符：确保列宽足够
            # 至少使用2个制表符来分隔列，如果列宽超过8的倍数则增加
            if max_len == 0:
                tabs_per_column.append(2)
            else:
                # 计算需要多少个制表符才能达到合适的列宽
                # 如果max_len <= 8，使用2个制表符
                # 如果max_len > 8，使用 (max_len // tab_width) + 2 个制表符
                if max_len <= tab_width:
                    tabs_per_column.append(2)
                else:
                    tabs_per_column.append((max_len // tab_width) + 2)
        
        # 生成表头
        header_line = ""
        for i, key in enumerate(column_keys):
            if i == 0:
                # 第一列没有标题，留空但保持对齐
                header_line += "".rjust(max_lengths[i])
            else:
                # 其他列右对齐标题
                header_line += headers[i].rjust(max_lengths[i])
            # 添加制表符
            header_line += "\t" * tabs_per_column[i]
        lines.append(header_line.rstrip())
        
        # 生成数据行
        for idx in range(len(column_data['model_name'])):
            row_line = ""
            for i, key in enumerate(column_keys):
                value = column_data[key][idx]
                # 右对齐
                row_line += str(value).rjust(max_lengths[i])
                # 添加制表符
                row_line += "\t" * tabs_per_column[i]
            lines.append(row_line.rstrip())
    else:
        # 当没有baseline模型时，也需要显示表头（包含新列）
        lines.append("                                 CV_R²              Test_R²              Test_R²_with_residual              Train_R²              training_time")
        lines.append("N/A                              N/A                  N/A                  null                  N/A                  N/A")
    
    lines.append("")
    
    # GP_Info 部分
    lines.append("GP_Info:")
    lines.append("")
    
    # 添加说明文字
    lines.append("Note: CV_Train shows 5-fold cross-validation results during evolution.")
    lines.append("      All_Train and All_Test show final model performance on the respective datasets without cross-validation.")
    lines.append("")
    
    # 准备表格数据
    cv_r2_mean = None
    cv_r2_std = None
    cv_mse_mean = None
    cv_mse_std = None
    cv_rmse_mean = None
    cv_rmse_std = None
    cv_mae_mean = None
    cv_mae_std = None
    
    # 添加CV_Train指标（如果存在）
    if cv_metrics:
        # 支持新格式 {folds: 5, cross_validation: {...}} 和旧格式 {folds: [...], summary: {...}}
        if 'cross_validation' in cv_metrics:
            # 新格式
            cross_validation = cv_metrics.get('cross_validation', {})
            cv_r2_mean = cross_validation.get('r2_mean', 0)
            cv_r2_std = cross_validation.get('r2_std', 0)
            cv_mse_mean = cross_validation.get('mse_mean', 0)
            cv_mse_std = cross_validation.get('mse_std', 0)
            cv_rmse_mean = cross_validation.get('rmse_mean', 0)
            cv_rmse_std = cross_validation.get('rmse_std', 0)
            cv_mae_mean = cross_validation.get('mae_mean', 0)
            cv_mae_std = cross_validation.get('mae_std', 0)
        elif 'summary' in cv_metrics:
            # 旧格式（兼容）
            summary = cv_metrics.get('summary', {})
            cv_r2_mean = summary.get('r2_mean', 0)
            cv_r2_std = summary.get('r2_std', 0)
            cv_mse_mean = summary.get('mse_mean', 0)
            cv_mse_std = summary.get('mse_std', 0)
            cv_rmse_mean = summary.get('rmse_mean', 0)
            cv_rmse_std = summary.get('rmse_std', 0)
            cv_mae_mean = summary.get('mae_mean', 0)
            cv_mae_std = summary.get('mae_std', 0)
    
    # 获取Train和Test指标
    train_r2 = train_metrics.get('r2', 0)
    train_mse = train_metrics.get('mse', 0)
    train_rmse = train_metrics.get('rmse', 0)
    train_mae = train_metrics.get('mae', 0)
    
    test_r2 = test_metrics.get('r2', 0)
    test_mse = test_metrics.get('mse', 0)
    test_rmse = test_metrics.get('rmse', 0)
    test_mae = test_metrics.get('mae', 0)
    
    # 收集所有列的数据
    gp_column_data = {
        'row_label': [],
        'r2': [],
        'mse': [],
        'rmse': [],
        'mae': []
    }
    
    # CV_Train行（如果有数据）
    if cv_r2_mean is not None:
        cv_r2_str = f"{cv_r2_mean:.6f}±{cv_r2_std:.6f}"
        cv_mse_str = f"{cv_mse_mean:.6f}±{cv_mse_std:.6f}"
        cv_rmse_str = f"{cv_rmse_mean:.6f}±{cv_rmse_std:.6f}"
        cv_mae_str = f"{cv_mae_mean:.6f}±{cv_mae_std:.6f}"
        gp_column_data['row_label'].append('CV_Train:')
        gp_column_data['r2'].append(cv_r2_str)
        gp_column_data['mse'].append(cv_mse_str)
        gp_column_data['rmse'].append(cv_rmse_str)
        gp_column_data['mae'].append(cv_mae_str)
    else:
        gp_column_data['row_label'].append('CV_Train:')
        gp_column_data['r2'].append('N/A')
        gp_column_data['mse'].append('N/A')
        gp_column_data['rmse'].append('N/A')
        gp_column_data['mae'].append('N/A')
    
    # All_Train行
    train_r2_str = f"{train_r2:.6f}"
    train_mse_str = f"{train_mse:.6f}"
    train_rmse_str = f"{train_rmse:.6f}"
    train_mae_str = f"{train_mae:.6f}"
    gp_column_data['row_label'].append('All_Train:')
    gp_column_data['r2'].append(train_r2_str)
    gp_column_data['mse'].append(train_mse_str)
    gp_column_data['rmse'].append(train_rmse_str)
    gp_column_data['mae'].append(train_mae_str)
    
    # All_Test行
    test_r2_str = f"{test_r2:.6f}"
    test_mse_str = f"{test_mse:.6f}"
    test_rmse_str = f"{test_rmse:.6f}"
    test_mae_str = f"{test_mae:.6f}"
    gp_column_data['row_label'].append('All_Test:')
    gp_column_data['r2'].append(test_r2_str)
    gp_column_data['mse'].append(test_mse_str)
    gp_column_data['rmse'].append(test_rmse_str)
    gp_column_data['mae'].append(test_mae_str)
    
    # 计算每列的最大长度（包括标题）
    gp_headers = ['', 'R2', 'MSE', 'RMSE', 'MAE']
    gp_column_keys = ['row_label', 'r2', 'mse', 'rmse', 'mae']
    
    gp_max_lengths = []
    for i, key in enumerate(gp_column_keys):
        # 第一列（行标签）没有标题，只从数据中计算最大长度
        max_len = len(gp_headers[i]) if i > 0 and gp_headers[i] else 0
        if key in gp_column_data:
            for value in gp_column_data[key]:
                max_len = max(max_len, len(str(value)))
        gp_max_lengths.append(max_len)
    
    # 计算每列需要的制表符数量（根据最大长度）
    tab_width = 8  # 制表符通常为8个字符宽度
    gp_tabs_per_column = []
    for max_len in gp_max_lengths:
        # 计算需要多少个制表符：确保列宽足够
        # 至少使用2个制表符来分隔列，如果列宽超过8的倍数则增加
        if max_len == 0:
            gp_tabs_per_column.append(2)
        else:
            # 计算需要多少个制表符才能达到合适的列宽
            # 如果max_len <= 8，使用2个制表符
            # 如果max_len > 8，使用 (max_len // tab_width) + 2 个制表符
            if max_len <= tab_width:
                gp_tabs_per_column.append(2)
            else:
                gp_tabs_per_column.append((max_len // tab_width) + 2)
    
    # 生成表头
    gp_header_line = ""
    for i, key in enumerate(gp_column_keys):
        if i == 0:
            # 第一列没有标题，留空但保持对齐
            gp_header_line += "".rjust(gp_max_lengths[i])
        else:
            # 其他列右对齐标题
            gp_header_line += gp_headers[i].rjust(gp_max_lengths[i])
        # 添加制表符
        gp_header_line += "\t" * gp_tabs_per_column[i]
    lines.append(gp_header_line.rstrip())
    
    # 生成数据行
    for idx in range(len(gp_column_data['row_label'])):
        row_line = ""
        for i, key in enumerate(gp_column_keys):
            value = gp_column_data[key][idx]
            # 右对齐
            row_line += str(value).rjust(gp_max_lengths[i])
            # 添加制表符
            row_line += "\t" * gp_tabs_per_column[i]
        lines.append(row_line.rstrip())
    lines.append(f"Expression: {expression}")
    # 添加Ridge公式
    if ridge_formula:
        lines.append(f"RidgeCV Formula: {ridge_formula}")
    
    # 添加训练时间（优先使用JSON中的真实训练时间）
    final_training_duration = None
    if json_gp_training_duration is not None:
        final_training_duration = json_gp_training_duration
    elif training_duration is not None:
        final_training_duration = training_duration
    
    if final_training_duration is not None:
        lines.append(f"Training Time: {final_training_duration:.4f} 秒 ({final_training_duration/60:.2f} 分钟)")
    
    # 添加残差模型训练时间（如果提供了）
    if residual_training_duration is not None:
        lines.append(f"Residual Model Training Time: {residual_training_duration:.4f} 秒 ({residual_training_duration/60:.2f} 分钟)")
    
    # 如果开启了残差拟合，显示残差拟合后的结果（放在RidgeCV Formula后面）
    if residual_fitting_results and isinstance(residual_fitting_results, dict):
        with_residual = residual_fitting_results.get('with_residual', {})
        # 支持新格式（列表）和旧格式（字典）
        if isinstance(with_residual, list) and len(with_residual) > 0:
            # 新格式：列表，遍历所有残差模型
            for residual_item in with_residual:
                residual_model_name = residual_item.get('model_name', 'Unknown')
                residual_train = residual_item.get('train', {})
                residual_test = residual_item.get('test', {})
                
                if residual_train and residual_test:
                    lines.append("")  # 空行分隔
                    lines.append(f"With Residual Fitting ({residual_model_name}):")
                    
                    # 获取IPS值
                    ips_r2_train = residual_train.get('ips_r2')
                    ips_sse_train = residual_train.get('ips_sse')
                    ips_r2_test = residual_test.get('ips_r2')
                    ips_sse_test = residual_test.get('ips_sse')
                    
                    # 构建训练集输出行
                    train_line = f"All_Train_Residual: R2: {residual_train.get('r2', 0):.6f}, MSE: {residual_train.get('mse', 0):.6f}, RMSE: {residual_train.get('rmse', 0):.6f}, MAE: {residual_train.get('mae', 0):.6f}"
                    if ips_r2_train is not None and ips_sse_train is not None:
                        train_line += f", IPS_R2: {ips_r2_train:.6f}, IPS_SSE: {ips_sse_train:.6f}"
                    lines.append(train_line)
                    
                    # 构建测试集输出行
                    test_line = f"All_Test_Residual:  R2: {residual_test.get('r2', 0):.6f}, MSE: {residual_test.get('mse', 0):.6f}, RMSE: {residual_test.get('rmse', 0):.6f}, MAE: {residual_test.get('mae', 0):.6f}"
                    if ips_r2_test is not None and ips_sse_test is not None:
                        test_line += f", IPS_R2: {ips_r2_test:.6f}, IPS_SSE: {ips_sse_test:.6f}"
                    lines.append(test_line)
        elif with_residual and isinstance(with_residual, dict):
            # 旧格式：字典
            residual_train = with_residual.get('train', {})
            residual_test = with_residual.get('test', {})
            residual_model = with_residual.get('residual_model')
            
            # 获取残差模型名称
            residual_model_name = "RandomForest"
            if residual_model is not None:
                try:
                    class_name = residual_model.__class__.__name__
                    if 'RandomForest' in class_name:
                        residual_model_name = "RandomForest"
                    elif 'Ridge' in class_name:
                        residual_model_name = "RidgeCV"
                except:
                    pass
            
            if residual_train and residual_test:
                lines.append("")  # 空行分隔
                lines.append(f"With Residual Fitting ({residual_model_name}):")
                
                # 获取IPS值
                ips_r2_train = residual_train.get('ips_r2')
                ips_sse_train = residual_train.get('ips_sse')
                ips_r2_test = residual_test.get('ips_r2')
                ips_sse_test = residual_test.get('ips_sse')
                
                # 构建训练集输出行
                train_line = f"All_Train_Residual: R2: {residual_train.get('r2', 0):.6f}, MSE: {residual_train.get('mse', 0):.6f}, RMSE: {residual_train.get('rmse', 0):.6f}, MAE: {residual_train.get('mae', 0):.6f}"
                if ips_r2_train is not None and ips_sse_train is not None:
                    train_line += f", IPS_R2: {ips_r2_train:.6f}, IPS_SSE: {ips_sse_train:.6f}"
                lines.append(train_line)
                
                # 构建测试集输出行
                test_line = f"All_Test_Residual:  R2: {residual_test.get('r2', 0):.6f}, MSE: {residual_test.get('mse', 0):.6f}, RMSE: {residual_test.get('rmse', 0):.6f}, MAE: {residual_test.get('mae', 0):.6f}"
                if ips_r2_test is not None and ips_sse_test is not None:
                    test_line += f", IPS_R2: {ips_r2_test:.6f}, IPS_SSE: {ips_sse_test:.6f}"
                lines.append(test_line)
    lines.append(f"Run time: {timestamp}")
    if evolution_json_path:
        lines.append(f"Evolution JSON: {evolution_json_path}")
        
        # 解析动态扩展日志，按代分组
        expansion_logs_by_gen = {}  # {generation: [log_lines]}
        if dynamic_expansion_logs:
            import re
            for expansion_log in dynamic_expansion_logs:
                # 从日志中提取代信息，例如 "第5代：触发自适应扩展..."
                match = re.search(r'第(\d+)代', expansion_log)
                if match:
                    gen_num = int(match.group(1))
                    if gen_num not in expansion_logs_by_gen:
                        expansion_logs_by_gen[gen_num] = []
                    # 将多行日志按行分割
                    for log_line in expansion_log.split('\n'):
                        if log_line.strip():  # 只添加非空行
                            expansion_logs_by_gen[gen_num].append(log_line.strip())
        
        # 读取每一代的进化信息并以与日志相同的格式输出
        try:
            if os.path.exists(evolution_json_path):
                with open(evolution_json_path, 'r', encoding='utf-8') as jf:
                    evo = _json_reader.load(jf)
                
                # 适配新结构：从gp_info.generations获取，如果没有则尝试旧结构generations
                if 'gp_info' in evo and 'generations' in evo['gp_info']:
                    gens = evo['gp_info']['generations']
                else:
                    gens = evo.get('generations', [])
                
                # 进化过程统计表格（与提供的格式完全一致）
                lines.append(
                    "\t\t\t\t\t                    fitness                    \t\t\t\t           size_tree                   "
                )
                lines.append(
                    "\t\t\t\t\t-----------------------------------------------\t-----------------------------------------------"
                )
                lines.append(
                    "gen\tnevals\tavg    \tgen\tmax    \tmin    \tnevals\tstd    \tavg \tgen\tmax\tmin\tnevals\tstd    "
                )
                for g in gens:
                    gen_idx = g.get('generation')
                    
                    # 在generation 0之前显示初始化种群信息
                    if gen_idx == 0 and population_init_info:
                        lines.append("")
                        # LLM和random特征统计
                        llm_count = population_init_info.get('llm_count', 0)
                        random_count = population_init_info.get('random_count', 0)
                        if llm_count > 0 or random_count > 0:
                            lines.append(f"初始化种群: 采用 {llm_count} 个LLM特征, 生成 {random_count} 个random特征")
                        
                        # High函数使用统计
                        high_usage = population_init_info.get('high_usage', {})
                        if high_usage:
                            for high_name in sorted(high_usage.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
                                count = high_usage[high_name]
                                lines.append(f"初始化后{high_name}的个体有 {count} 个")
                        lines.append("")
                    
                    # 在当前代之前显示对应的动态扩展日志
                    if gen_idx in expansion_logs_by_gen:
                        lines.append("")
                        for log_line in expansion_logs_by_gen[gen_idx]:
                            lines.append(log_line)
                        lines.append("")
                    
                    stats = g.get('statistics', {}) or {}
                    fit = stats.get('fitness', {}) if isinstance(stats.get('fitness', {}), dict) else {}
                    siz = stats.get('size_tree', {}) if isinstance(stats.get('size_tree', {}), dict) else {}
                    def _fmt(x):
                        try:
                            return f"{float(x):.6f}"
                        except Exception:
                            return "0.000000"
                    # JSON里未保存nevals，统一按0输出以保持列位
                    nevals = 0
                    line = (
                        f"{gen_idx}\t{nevals}\t{_fmt(fit.get('avg'))}\t{gen_idx}\t{_fmt(fit.get('max'))}\t{_fmt(fit.get('min'))}\t{nevals}\t{_fmt(fit.get('std'))}\t"
                        f"{_fmt(siz.get('avg'))}\t{gen_idx}\t{_fmt(siz.get('max'))}\t{_fmt(siz.get('min'))}\t{nevals}\t{_fmt(siz.get('std'))}"
                    )
                    lines.append(line)
                    
                    # 添加每一代学习到的最佳特征表达式（适配新结构：whole_individuals）
                    whole_individuals = g.get('whole_individuals', [])
                    if whole_individuals and len(whole_individuals) > 0:
                        best_expr = whole_individuals[0].get('gp_expression', '')
                        if best_expr:
                            lines.append(f"{gen_idx} {best_expr}")
                    else:
                        # 兼容旧结构：best_individuals
                        best_individuals = g.get('best_individuals', [])
                        if best_individuals and len(best_individuals) > 0:
                            best_expr = best_individuals[0].get('expression', '')
                            if best_expr:
                                lines.append(f"{gen_idx} {best_expr}")
                # 追加一空行与日志风格保持分段
                lines.append("")
        except Exception as e:
            lines.append(f"[WARN] 读取进化信息失败: {e}")
    lines.append("=" * 80)

    content = "\n".join(lines) + "\n\n"
    with open(evo_file, "a", encoding="utf-8") as f:
        f.write(content)
    return evo_file

