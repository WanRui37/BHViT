import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===================== 0. 全局绘图风格设置 =====================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.linewidth": 1.5,  # 坐标轴线宽
        "xtick.major.width": 1.2,  # 刻度线宽
        "ytick.major.width": 1.2,
        "xtick.direction": "in",  # 刻度向内
        "ytick.direction": "in",
        "mathtext.fontset": "stix",  # 公式字体风格
    }
)


# ===================== 1. Accuracy–Compute trade-off 图表 =====================
def plot_accuracy_compute_tradeoff():
    """
    绘制Accuracy–Compute trade-off图（图A）
    x轴：OPs（×10^8），使用对数坐标
    y轴：Top-1 Accuracy（%）
    点的颜色：按模型类别分组（Transformer/ViT 和 CNN/BNN）
    点的形状：按精度 W-A（如 32-32 / 2-2 / 1-1 / 8-8 / 4-4）
    """

    # 创建示例数据（实际使用时应替换为真实数据）
    # 数据格式：模型名, 类别, W-A精度, OPs(×10^8), Top-1 Accuracy(%)
    data = [
        # Transformer/ViT 类别
        ("DeiT-Small", "Transformer", "32-32", 46.0, 70.56),
        ("Q-ViT-T 4-bit", "Transformer", "4-4", 1.41, 74.3),
        ("Q-ViT-S 2-bit", "Transformer", "2-2", 2.83, 68.30),
        ("Q-ViT-S 1-bit", "Transformer", "1-1", 1.41, 50.26),
        ("GSB-ViT-S 1-bit", "Transformer", "1-1", 1.68, 71.10),
        ("BHViT-T", "Transformer", "4-4", 1.56, 75.75),
        ("BHViT-S", "Transformer", "1-1", 1.56, 75.17),
        # CNN/BNN 类别
        ("ResNet-18", "CNN", "32-32", 18.1, 72.50),
        ("XNOR-Net-18", "CNN", "1-1", 1.67, 53.76),
        ("Bi-RealNet-18", "CNN", "1-1", 1.63, 54.34),
        ("ReActNet-A", "CNN", "1-1", 0.87, 50.30),
    ]

    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=["Model", "Category", "Precision", "OPs", "Accuracy"])

    # ===================== 2. 聚类分析 =====================
    # 提取特征用于聚类 (Accuracy, OPs)
    features = df[["Accuracy", "OPs"]].values

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 使用K-means聚类将模型分为3个簇
    # 当数据点少于3个时，调整聚类数量
    n_clusters = min(3, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # 将聚类结果添加到数据框中
    df["Cluster"] = clusters

    # 定义颜色映射
    category_colors = {"Transformer": "#1f77b4", "CNN": "#ff7f0e"}  # 蓝色  # 橙色

    # 定义标记映射
    precision_markers = {
        "32-32": "o",  # 圆点
        "1-1": "^",  # 三角
        "2-2": "s",  # 方块
        "8-8": "D",  # 菱形
        "4-4": "v",  # 倒三角
        "3-3": "<",  # 左三角
    }

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 定义每个簇的放大因子
    scale_factors = {0: 2.0, 1: 3.0, 2: 4.0}  # 可根据需要调整每个簇的放大因子

    # 为每个聚类绘制矩形区域
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 为聚类定义颜色
    for cluster_id in range(n_clusters):
        # 获取当前聚类的所有点
        cluster_points = df[df["Cluster"] == cluster_id][["OPs", "Accuracy"]].values

        # 只有当簇中有足够多的点时才绘制矩形
        if len(cluster_points) > 1:
            # 计算边界框
            min_ops = np.min(cluster_points[:, 0])
            max_ops = np.max(cluster_points[:, 0])
            min_acc = np.min(cluster_points[:, 1])
            max_acc = np.max(cluster_points[:, 1])

            # 计算中心点
            center_ops = (min_ops + max_ops) / 2
            center_acc = (min_acc + max_acc) / 2

            # 计算宽度和高度
            width_ops = max_ops - min_ops
            height_acc = max_acc - min_acc

            # 使用为每个簇定义的放大因子
            scale_factor = scale_factors.get(cluster_id, 3.0)  # 默认值为3.0
            width_ops *= scale_factor
            height_acc *= scale_factor

            # 调整边界以考虑放大因子
            min_ops = center_ops - width_ops / 2
            max_ops = center_ops + width_ops / 2
            min_acc = center_acc - height_acc / 2
            max_acc = center_acc + height_acc / 2

            # 确保矩形不会超出合理的范围
            min_ops = max(min_ops, np.min(df["OPs"]) / 2)
            max_ops = min(max_ops, np.max(df["OPs"]) * 2)
            min_acc = max(min_acc, np.min(df["Accuracy"]) - 5)
            max_acc = min(max_acc, np.max(df["Accuracy"]) + 5)

            # 创建矩形阴影
            from matplotlib.patches import Rectangle

            rectangle = Rectangle(
                (min_ops, min_acc),
                max_ops - min_ops,
                max_acc - min_acc,
                facecolor=colors[cluster_id % len(colors)],
                alpha=0.2,
                zorder=1,
            )
            ax.add_patch(rectangle)

    # 绘制散点图
    for category in df["Category"].unique():
        for precision in df["Precision"].unique():
            subset = df[(df["Category"] == category) & (df["Precision"] == precision)]
            if not subset.empty:
                ax.scatter(
                    subset["OPs"],
                    subset["Accuracy"],
                    c=category_colors[category],
                    marker=precision_markers[precision],
                    s=100,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"{category} ({precision})" if precision in ["32-32", "1-1", "2-2", "8-8", "4-4"] else None,
                )

                # 为所有点添加标注
                for _, row in subset.iterrows():
                    if row["Model"] in ["Q-ViT-T 4-bit", "IR-Net", "DeiT-Small"]:
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(-10, -5),  # 左下角偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="right",  # 水平右对齐
                            va="top",  # 垂直顶部对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )
                    elif row["Model"] in ["ReActNet-A"]:
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(-20, 22),  # 左下角偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",  # 水平右对齐
                            va="bottom",  # 垂直顶部对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )
                    elif row["Model"] in ["XNOR-Net-18"]:
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(10, 17),  # 左下角偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",  # 水平右对齐
                            va="bottom",  # 垂直顶部对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )
                    elif row["Model"] in ["Bi-RealNet-18"]:
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(-50, 40),  # 左下角偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",  # 水平右对齐
                            va="top",  # 垂直顶部对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )
                    elif row["Model"] in ["BHViT-S"]:
                        # BHViT模型位于高精度区域，标注放在左侧避免重叠
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(-15, 5),  # 左侧偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="right",  # 水平右对齐
                            va="center",  # 垂直居中对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )
                    elif row["Model"] in ["BHViT-T"]:
                        # BHViT模型位于高精度区域，标注放在左侧避免重叠
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(20, 5),  # 左侧偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",  # 水平右对齐
                            va="center",  # 垂直居中对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )
                    else:
                        ax.annotate(
                            row["Model"],
                            (row["OPs"], row["Accuracy"]),
                            xytext=(5, 5),  # 默认右上角偏移
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",  # 水平左对齐
                            va="bottom",  # 垂直底部对齐
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),  # 添加连线
                        )

    # 设置坐标轴
    ax.set_xscale("log")
    ax.set_xlabel("OPs ($\\times10^8$)", fontsize=14)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14)

    # 设置网格
    ax.grid(True, linestyle="--", alpha=0.7)

    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    # 只保留主要类别的图例
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label and label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc="lower right", fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig("accuracy_compute_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("accuracy_compute_tradeoff.png", dpi=300, bbox_inches="tight")


# ===================== 2. 主函数 =====================
if __name__ == "__main__":
    # 绘制Accuracy–Compute trade-off图
    plot_accuracy_compute_tradeoff()
