"""
主程序入口：划分数据集、生成 KS 检验 PDF（单图多子图，dpi=300），并将 PDF 路径写入 JSON。
"""

from utils.dataset_splitter import DatasetSplitter
from utils.generate_ks_plots import generate_ks_plots_pdf

# 配置参数
dataset_path = "ash_fusion_dataset_cleaned.csv"
target_columns = ["Ash_Deformation", "Ash_Softening", "Ash_Fluid"]
delete_columns = ["Sample_ID"]
train_ratio = 0.8
random_seed = 43

splitter = DatasetSplitter(
    dataset_path=dataset_path,
    target_columns=target_columns,
    delete_columns=delete_columns,
    train_ratio=train_ratio,
    random_seed=random_seed,
)

result = splitter.split_dataset()
if result:
    pdf_abs = generate_ks_plots_pdf(
        result["train_df"],
        result["test_df"],
        result["feature_columns"],
        result["target_columns"],
        result["abs_output_folder"],
    )
    DatasetSplitter.update_config_json_ks_path(result["config_abs_path"], pdf_abs)
    print(f"已更新 JSON 中的 ks_plots_pdf_path: {pdf_abs}")
