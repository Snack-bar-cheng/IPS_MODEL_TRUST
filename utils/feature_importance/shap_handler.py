"""
基于SHAP的特征重要性计算工具
"""

import os
import time
from typing import List, Optional, Tuple

import numpy as np


def _is_tree_model(model) -> bool:
    """简单判断是否为树模型（用于优先使用TreeExplainer）。"""
    return hasattr(model, "feature_importances_")


def _normalize_feature_names(feature_names: Optional[List[str]], n_features: int) -> List[str]:
    """确保特征名称长度与特征数一致。"""
    if not feature_names or len(feature_names) != n_features:
        return [f"feature_{i}" for i in range(n_features)]
    return list(feature_names)


def _is_dnn_model(model) -> bool:
    """简单判断是否为DNNRegressor（带model与scaler属性）。"""
    return hasattr(model, "model") and hasattr(model, "scaler")


def _to_2d_array(values):
    """将shap_values转换为二维数组 (n_samples, n_features)。"""
    if isinstance(values, list):
        values = values[0]
    arr = np.array(values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _to_interaction_array(values):
    """将交互shap值转换为三维数组 (n_samples, n_features, n_features)。"""
    if isinstance(values, list):
        values = values[0]
    return np.array(values)


def compute_shap_importances(
    model,
    X_train,
    X_explain,
    feature_names: Optional[List[str]] = None,
    target_name: str = "",
    model_name: str = "",
    random_seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    prefix: str = "",
    background_limit: Optional[int] = None,
    explain_limit: Optional[int] = None,
) -> Tuple[Optional[List[dict]], Optional[List[dict]], Optional[str], Optional[str]]:
    """
    计算SHAP特征重要性与交互重要性，并生成summary图。
    
    返回:
        feature_importances: [{"mean_abs_shap": float, "mean_shap": float, "feature_name": str}, ...] 或 None
        interaction_importances: [{"mean_abs_shap": float, "mean_shap": float, "feature_name": [str, str]}, ...] 或 None
        shap_plot_path: 保存的summary图路径或None
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception:
        # shap未安装或matplotlib不可用时回退
        return None, None, None, None
    
    try:
        X_train = np.array(X_train)
        X_explain = np.array(X_explain)
        
        # 应用解释数据集限制（None表示不限制，使用全部数据）
        if explain_limit is None:
            X_explain_subset = X_explain
        else:
            X_explain_subset = X_explain[:min(explain_limit, X_explain.shape[0])]
        
        shap_values = None
        interaction_importances = None
        
        # ===== 树模型：TreeExplainer =====
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain_subset)
            shap_values = _to_2d_array(shap_values)
        # ===== DNN模型：DeepExplainer =====
        elif _is_dnn_model(model):
            try:
                import torch
            except Exception as e:
                print(f"[SHAP][ERROR] DNN模型需要torch，但导入失败: {e}")
                return None, None, None, None
            
            device = getattr(model, "device", None)
            scaler = getattr(model, "scaler", None)
            net = getattr(model, "model", None)
            
            print(f"[SHAP][DEBUG] DNN模型检查: device={device}, scaler={scaler is not None}, net={net is not None}")
            
            if scaler is None or net is None:
                print(f"[SHAP][ERROR] DNN模型缺少必要属性: scaler={scaler is not None}, net={net is not None}")
                return None, None, None, None
            
            device = device if device is not None else torch.device("cpu")
            print(f"[SHAP][INFO] 使用设备进行SHAP计算: {device}")
            
            try:
                X_train_f = scaler.transform(X_train).astype(np.float32)
                X_explain_f = scaler.transform(X_explain_subset).astype(np.float32)
                
                # 应用背景数据集限制（None表示不限制，使用全部数据）
                if background_limit is None:
                    X_train_f_subset = X_train_f
                    bg_limit_actual = X_train_f.shape[0]
                else:
                    bg_limit_actual = min(background_limit, X_train_f.shape[0])
                    X_train_f_subset = X_train_f[:bg_limit_actual]
                
                # 应用解释数据集限制（None表示不限制，使用全部数据）
                if explain_limit is None:
                    X_explain_f_subset = X_explain_f
                    explain_limit_actual = X_explain_f.shape[0]
                else:
                    explain_limit_actual = min(explain_limit, X_explain_f.shape[0])
                    X_explain_f_subset = X_explain_f[:explain_limit_actual]
                    # 同步更新X_explain_subset用于绘图
                    X_explain_subset = X_explain_subset[:explain_limit_actual]
                
                # 确保数据在正确的设备上
                background = torch.tensor(X_train_f_subset, device=device, requires_grad=False)
                explain_tensor = torch.tensor(X_explain_f_subset, device=device, requires_grad=False)
                
                print(f"[SHAP][DEBUG] DNN SHAP计算: device={device}, background_shape={background.shape}, explain_shape={explain_tensor.shape}, background_limit={background_limit}, explain_limit={explain_limit}")
                print(f"[SHAP][DEBUG] background设备: {background.device}, explain_tensor设备: {explain_tensor.device}")
                
                net.eval()
                # 确保网络在正确的设备上
                net = net.to(device)
                
                def model_forward(x):
                    """PyTorch模型前向传播函数（支持梯度计算，在正确设备上）"""
                    # 确保输入在正确的设备上
                    if isinstance(x, torch.Tensor):
                        if x.device != device:
                            x = x.to(device)
                        x = x.requires_grad_(True)
                    else:
                        # 如果是numpy数组，转换为tensor并移到正确设备
                        x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=True)
                    
                    out = net(x)
                    if out.ndim == 1:
                        out = out.unsqueeze(-1)
                    return out
                
                print(f"[SHAP][INFO] 正在创建Explainer（适用于PyTorch模型，设备={device}）...")
                # 对于PyTorch模型，使用通用的Explainer，它会自动选择合适的方法
                # SHAP计算通常在CPU上进行，但模型前向传播可以在GPU/MPS上
                # 将数据移到CPU用于SHAP计算，但model_forward会在原始设备上计算
                background_cpu = background.cpu().numpy()
                explain_tensor_cpu = explain_tensor.cpu().numpy()
                
                masker = shap.maskers.Independent(background_cpu, max_samples=bg_limit_actual)
                explainer = shap.Explainer(model_forward, masker)
                print(f"[SHAP][INFO] 正在计算DNN的SHAP值（模型在{device}上，SHAP计算在CPU上，这可能需要一些时间）...")
                shap_explanation = explainer(explain_tensor_cpu)
                print(f"[SHAP][INFO] DNN SHAP值计算完成")
                
                # 从Explanation对象中提取values
                if hasattr(shap_explanation, 'values'):
                    shap_values = shap_explanation.values
                else:
                    shap_values = shap_explanation
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = np.array(shap_values)
                if shap_values.ndim == 1:
                    shap_values = shap_values.reshape(1, -1)
            except Exception as e:
                print(f"[SHAP][ERROR] DNN SHAP计算失败: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, None
        else:
            return None, None, None, None
        
        shap_values = _to_2d_array(shap_values)
        n_features = shap_values.shape[1]
        names = _normalize_feature_names(feature_names, n_features)
        
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        mean_signed = np.mean(shap_values, axis=0)
        
        feature_importances = [
            {
                "mean_abs_shap": float(ma),
                "mean_shap": float(ms),
                "feature_name": names[i]
            }
            for i, (ma, ms) in enumerate(zip(mean_abs, mean_signed))
        ]
        feature_importances.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        
        # 交互重要性：仅树模型尝试，DNN置None
        if _is_tree_model(model):
            try:
                interaction_values = explainer.shap_interaction_values(X_explain_subset)
                interaction_values = _to_interaction_array(interaction_values)
                if interaction_values.ndim == 3:
                    mean_abs_inter = np.mean(np.abs(interaction_values), axis=0)
                    mean_signed_inter = np.mean(interaction_values, axis=0)
                    
                    inter_list = []
                    for i in range(n_features):
                        for j in range(i + 1, n_features):
                            inter_list.append({
                                "mean_abs_shap": float(mean_abs_inter[i, j]),
                                "mean_shap": float(mean_signed_inter[i, j]),
                                "feature_name": [names[i], names[j]]
                            })
                    if inter_list:
                        inter_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
                        interaction_importances = inter_list
            except Exception:
                interaction_importances = None
        
        shap_plot_path = None
        shap_bee_path = None
        try:
            plt.rcParams.update({"font.family": "Times New Roman"})
            shap.summary_plot(
                shap_values,
                X_explain_subset,
                feature_names=names,
                show=False,
                plot_type="bar"
            )
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ts = int(time.time())
                seed_str = str(random_seed) if random_seed is not None else "seed"
                prefix_str = f"{prefix}_" if prefix else ""
                filename_bar = f"{target_name or 'target'}_{model_name or 'model'}_{seed_str}_{ts}_{prefix_str}bar.png"
                shap_plot_path = os.path.join(save_dir, filename_bar)
                plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # 额外的SHAP散点图（beeswarm）
            shap.summary_plot(
                shap_values,
                X_explain_subset,
                feature_names=names,
                show=False,
                plot_type="dot"
            )
            if save_dir:
                ts2 = int(time.time())
                prefix_str = f"{prefix}_" if prefix else ""
                filename_bee = f"{target_name or 'target'}_{model_name or 'model'}_{seed_str}_{ts2}_{prefix_str}bee.png"
                shap_bee_path = os.path.join(save_dir, filename_bee)
                plt.savefig(shap_bee_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            shap_plot_path = None
            shap_bee_path = None
        
        return feature_importances, interaction_importances, shap_plot_path, shap_bee_path
    except Exception as e:
        print(f"[SHAP][ERROR] compute_shap_importances发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


