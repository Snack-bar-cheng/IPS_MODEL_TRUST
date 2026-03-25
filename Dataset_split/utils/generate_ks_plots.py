"""
生成 KS 检验图：单 PDF、多子图、KDE 曲线；供 main 与可选批处理使用。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

KS_PDF_DPI = 300

plt.rcParams["font.sans-serif"] = ["SimSun", "STSong", "SimHei", "Microsoft YaHei"]
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def convert_to_subscript(text):
    subscript_map = {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    }
    result = ""
    for char in text:
        if char.isdigit():
            result += subscript_map.get(char, char)
        else:
            result += char
    return result


def plot_single_subplot(ax, train_data, test_data, feature_name, alpha=0.05, show_ylabel=False):
    train_clean = train_data.dropna()
    test_clean = test_data.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        ax.text(0.5, 0.5, f"{feature_name}\n数据为空",
                transform=ax.transAxes, ha="center", va="center")
        return None, None

    statistic, p_value = stats.ks_2samp(train_clean, test_clean)

    n_bins = min(50, int(np.sqrt(len(train_clean) + len(test_clean))))

    ax.hist(train_clean, bins=n_bins, alpha=0.5,
            color="blue", density=True, edgecolor="black", linewidth=0.5)
    ax.hist(test_clean, bins=n_bins, alpha=0.5,
            color="red", density=True, edgecolor="black", linewidth=0.5)

    try:
        from scipy.stats import gaussian_kde

        all_values = np.concatenate([train_clean, test_clean])
        x_min, x_max = all_values.min(), all_values.max()
        x_range = np.linspace(x_min, x_max, 1000)

        kde_train = gaussian_kde(train_clean)
        kde_test = gaussian_kde(test_clean)

        ax.plot(x_range, kde_train(x_range),
                linewidth=2, color="darkblue", linestyle="--")
        ax.plot(x_range, kde_test(x_range),
                linewidth=2, color="darkred", linestyle="--")
    except Exception as e:
        print(f"警告: 计算KDE时出错: {str(e)}")

    feature_name_subscript = convert_to_subscript(feature_name)
    has_chinese = any("\u4e00" <= char <= "\u9fff" for char in feature_name)
    xlabel_font = "serif" if not has_chinese else "sans-serif"

    ax.set_xlabel(f"{feature_name_subscript}", fontsize=16, family=xlabel_font)

    if show_ylabel:
        ax.set_ylabel("Density", fontsize=16, family="serif")
    else:
        ax.set_ylabel("")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    for label in ax.get_xticklabels():
        label.set_fontfamily("serif")
        label.set_fontsize(15)
    for label in ax.get_yticklabels():
        label.set_fontfamily("serif")
        label.set_fontsize(15)

    test_result = "Pass" if p_value >= alpha else "Fail"
    info_text = f"KS = {statistic:.4f}\np = {p_value:.4f}\n{test_result} (α={alpha})"

    ax.text(0.98, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="#F8F9FA", alpha=0.9,
                      edgecolor="#DEE2E6", linewidth=1.5),
            family="serif")

    return statistic, p_value


def _ordered_feature_list(feature_columns):
    cols = list(feature_columns)
    if "MnO" in cols and "TiO2" in cols:
        tio2_idx = cols.index("TiO2")
        mno_idx = cols.index("MnO")
        if mno_idx != tio2_idx + 1:
            cols.remove("MnO")
            cols.insert(tio2_idx + 1, "MnO")
    return cols


def plot_all_features_pdf(train_df, test_df, feature_names, output_path, alpha=0.05, dpi=KS_PDF_DPI):
    """
    所有变量单张大图（多行多列子图），输出 PDF，dpi 默认 300。
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), squeeze=False)

    results = []
    for idx, feature_name in enumerate(feature_names):
        row = idx // n_cols
        col = idx % n_cols
        show_ylabel = col == 0

        statistic, p_value = plot_single_subplot(
            axes[row, col],
            train_df[feature_name],
            test_df[feature_name],
            feature_name,
            alpha,
            show_ylabel=show_ylabel,
        )
        if statistic is not None:
            results.append((feature_name, statistic, p_value))

    total_subplots = n_rows * n_cols
    if n_features < total_subplots:
        last_empty_idx = n_features
        row = last_empty_idx // n_cols
        col = last_empty_idx % n_cols
        ax_legend = axes[row, col]
        ax_legend.axis("off")

        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)

        ax_legend.text(0.5, 0.95, "Legend",
                       transform=ax_legend.transAxes,
                       fontsize=16, fontweight="bold",
                       horizontalalignment="center",
                       family="serif")

        y_pos = 0.80
        rect_train = Rectangle((0.1, y_pos - 0.03), 0.15, 0.06,
                               facecolor="blue", alpha=0.5, edgecolor="black", linewidth=0.5)
        ax_legend.add_patch(rect_train)
        ax_legend.text(0.3, y_pos, "Training set",
                       transform=ax_legend.transAxes,
                       fontsize=13, verticalalignment="center",
                       family="serif")

        y_pos = 0.70
        rect_test = Rectangle((0.1, y_pos - 0.03), 0.15, 0.06,
                              facecolor="red", alpha=0.5, edgecolor="black", linewidth=0.5)
        ax_legend.add_patch(rect_test)
        ax_legend.text(0.3, y_pos, "Test set",
                       transform=ax_legend.transAxes,
                       fontsize=13, verticalalignment="center",
                       family="serif")

        y_pos = 0.55
        line_train = Line2D([0.1, 0.25], [y_pos, y_pos],
                            color="darkblue", linestyle="--", linewidth=2.5)
        ax_legend.add_line(line_train)
        ax_legend.text(0.3, y_pos, "Training set KDE",
                       transform=ax_legend.transAxes,
                       fontsize=13, verticalalignment="center",
                       family="serif")

        y_pos = 0.45
        line_test = Line2D([0.1, 0.25], [y_pos, y_pos],
                           color="darkred", linestyle="--", linewidth=2.5)
        ax_legend.add_line(line_test)
        ax_legend.text(0.3, y_pos, "Test set KDE",
                       transform=ax_legend.transAxes,
                       fontsize=13, verticalalignment="center",
                       family="serif")

        y_pos = 0.30
        ax_legend.text(0.1, y_pos, "KS test:",
                       transform=ax_legend.transAxes,
                       fontsize=13, fontweight="bold",
                       family="serif")
        y_pos = 0.20
        ax_legend.text(0.15, y_pos, "• Pass: p ≥ α (α = 0.05)",
                       transform=ax_legend.transAxes,
                       fontsize=12, verticalalignment="center",
                       family="serif")
        y_pos = 0.12
        ax_legend.text(0.15, y_pos, "• Fail: p < α",
                       transform=ax_legend.transAxes,
                       fontsize=12, verticalalignment="center",
                       family="serif")

        for idx in range(n_features + 1, total_subplots):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")
    else:
        for idx in range(n_features, total_subplots):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"已生成 KS PDF: {output_path} (共 {n_features} 个子图, dpi={dpi})")
    return results


def generate_summary_report(train_df, test_df, feature_columns, output_dir, config_name):
    results = []

    for feature in feature_columns:
        train_clean = train_df[feature].dropna()
        test_clean = test_df[feature].dropna()

        if len(train_clean) > 0 and len(test_clean) > 0:
            statistic, p_value = stats.ks_2samp(train_clean, test_clean)
            results.append({
                "变量": feature,
                "KS统计量": round(statistic, 6),
                "p值": round(p_value, 6),
                "是否通过(α=0.05)": "是" if p_value >= 0.05 else "否",
            })

    results_df = pd.DataFrame(results)
    out_csv = Path(output_dir) / "ks_test_summary.csv"
    results_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n汇总报告已保存: {out_csv}")
    print(f"共处理 {len(results)} 个变量")
    print(f"通过检验: {sum(1 for r in results if r['是否通过(α=0.05)'] == '是')} 个")
    print(f"未通过检验: {sum(1 for r in results if r['是否通过(α=0.05)'] == '否')} 个")


def generate_ks_plots_pdf(train_df, test_df, feature_columns, target_columns,
                          abs_output_folder, alpha=0.05, pdf_filename="ks_test_all_features.pdf"):
    """
    在 abs_output_folder/ks_plots 下生成单份 PDF（所有变量为子图），返回 PDF 绝对路径。
    feature_columns: 用于 KS 的特征名列表；target_columns 中存在于表中的列会追加到图中（与旧逻辑一致）。
    """
    ks_dir = Path(abs_output_folder) / "ks_plots"
    ks_dir.mkdir(parents=True, exist_ok=True)

    ordered = _ordered_feature_list(feature_columns)
    all_names = list(ordered)
    for t in target_columns:
        if t in train_df.columns and t not in all_names:
            all_names.append(t)

    pdf_path = ks_dir / pdf_filename
    abs_pdf = str(pdf_path.resolve())

    plot_all_features_pdf(train_df, test_df, all_names, abs_pdf, alpha=alpha, dpi=KS_PDF_DPI)

    config_name = Path(abs_output_folder).name
    generate_summary_report(train_df, test_df, ordered, ks_dir, config_name)

    return abs_pdf


def process_config_folder(config_folder_path):
    """
    批处理：对单个配置文件夹（含 train/test CSV）生成 KS PDF 与汇总 CSV。
    """
    config_folder = Path(config_folder_path)
    config_name = config_folder.name

    train_file = config_folder / f"trainset_{config_name}.csv"
    test_file = config_folder / f"testset_{config_name}.csv"

    if not train_file.exists() or not test_file.exists():
        print(f"警告: {config_name} 文件夹中缺少训练集或测试集文件")
        return None

    print(f"\n处理配置: {config_name}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    if list(train_df.columns) != list(test_df.columns):
        print(f"警告: {config_name} 训练集和测试集的列名不一致")
        return None

    target_columns = ["Ash_Deformation", "Ash_Softening", "Ash_Fluid"]
    feature_columns = [col for col in train_df.columns if col not in target_columns]

    pdf_abs = generate_ks_plots_pdf(
        train_df, test_df, feature_columns, target_columns,
        str(config_folder.resolve()),
    )
    return pdf_abs


def main_batch():
    """遍历 dataset_onfiguration 下各子文件夹并生成 KS PDF。"""
    base_dir = Path(__file__).resolve().parent.parent
    dataset_config = base_dir / "dataset_onfiguration"

    if not dataset_config.exists():
        print(f"错误: 目录不存在: {dataset_config}")
        return

    config_folders = [d for d in dataset_config.iterdir() if d.is_dir()]
    if not config_folders:
        print(f"警告: 在 {dataset_config} 中未找到配置文件夹")
        return

    print(f"找到 {len(config_folders)} 个配置文件夹")
    for config_folder in config_folders:
        try:
            process_config_folder(config_folder)
        except Exception as e:
            print(f"错误: 处理 {config_folder.name} 时出错: {str(e)}")

    print("\n所有 KS PDF 生成完成！")


if __name__ == "__main__":
    main_batch()
