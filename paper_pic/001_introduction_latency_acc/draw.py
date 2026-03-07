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



# ===================== 2. Accuracy–Latency trade-off 图表 =====================
def plot_accuracy_latency_tradeoff():
    """
    Plot Accuracy–Latency trade-off chart for specific models
    x-axis: Latency (ms/img)
    y-axis: Top-1 Accuracy (%)
    Point colors: by model type
    Point shapes: by precision W-A
    """

    # Data from the provided table
    # Format: Model, Precision, Params(MB), OPs(×10^8), Latency(ms/img), Speedup
    # Note: We need to add accuracy data for these models
    data = [
        # Model, Category, Precision, Latency(ms/img), Accuracy(%)
        ("BHViT-S", "Transformer", "1-1", 6.8, 72.37),  # Accuracy from previous data
        ("BHViT-T", "Transformer", "4-4", 5.3, 72.94),  # Accuracy from previous data
        ("GSB-ViT-S", "Transformer", "1-1", 5.7, 71.10),  # Accuracy from previous data
        ("Q-ViT-S", "Transformer", "1-1", 5.4, 50.26),  # Accuracy from previous data
        ("Q-ViT-T", "Transformer", "4-4", 4.1, 72.66),  # Accuracy from previous data
    ]

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=["Model", "Category", "Precision", "Latency", "Accuracy"])

    # Define color mapping
    category_colors = {"Transformer": "#1f77b4"}  # Blue for Transformer models

    # Define marker mapping
    precision_markers = {
        "1-1": "^",  # Triangle
        "4-4": "v",  # Inverted triangle
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define regions for rectangular areas
    # Upper region: higher accuracy area
    upper_region_x = [4, 7, 7, 4, 4]  # x coordinates for rectangle
    upper_region_y = [70, 70, 73.5, 73.5, 70]  # y coordinates for rectangle
    
    # Lower region: lower accuracy area  
    lower_region_x = [4, 7, 7, 4, 4]  # x coordinates for rectangle
    lower_region_y = [49, 49, 51, 51, 49]  # y coordinates for rectangle

    # Plot rectangular regions with different colors
    ax.fill(upper_region_x, upper_region_y, color='lightgreen', alpha=0.3, label='High Accuracy Region')
    ax.fill(lower_region_x, lower_region_y, color='lightcoral', alpha=0.3, label='Low Accuracy Region')

    # Plot scatter points
    for category in df["Category"].unique():
        for precision in df["Precision"].unique():
            subset = df[(df["Category"] == category) & (df["Precision"] == precision)]
            if not subset.empty:
                ax.scatter(
                    subset["Latency"],
                    subset["Accuracy"],
                    c=category_colors[category],
                    marker=precision_markers[precision],
                    s=120,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.8,
                    label=f"{category} ({precision})",
                )

                # Add annotations for all points
                for _, row in subset.iterrows():
                    if row["Model"] == "BHViT-S":
                        ax.annotate(
                            row["Model"],
                            (row["Latency"], row["Accuracy"]),
                            xytext=(-15, -10),  # Bottom-left offset
                            textcoords="offset points",
                            fontsize=11,
                            ha="right",
                            va="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        )
                    elif row["Model"] == "BHViT-T":
                        ax.annotate(
                            row["Model"],
                            (row["Latency"], row["Accuracy"]),
                            xytext=(-10, -10),  # Bottom-right offset
                            textcoords="offset points",
                            fontsize=11,
                            ha="right",
                            va="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        )
                    elif row["Model"] == "GSB-ViT-S":
                        ax.annotate(
                            row["Model"],
                            (row["Latency"], row["Accuracy"]),
                            xytext=(10, -10),  # Bottom-right offset
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",
                            va="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        )
                    elif row["Model"] == "Q-ViT-S":
                        ax.annotate(
                            row["Model"],
                            (row["Latency"], row["Accuracy"]),
                            xytext=(-15, 10),  # Top-left offset
                            textcoords="offset points",
                            fontsize=11,
                            ha="right",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        )
                    elif row["Model"] == "Q-ViT-T":
                        ax.annotate(
                            row["Model"],
                            (row["Latency"], row["Accuracy"]),
                            xytext=(15, -10),  # Top-right offset
                            textcoords="offset points",
                            fontsize=11,
                            ha="left",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        )

    # Set axes
    ax.set_xlabel("Latency (ms/img)", fontsize=14)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14)

    # Set grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set legend
    ax.legend(loc="lower right", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save image
    plt.savefig("accuracy_latency_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("accuracy_latency_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.show()


# ===================== 3. Main function =====================
if __name__ == "__main__":
    # Plot Accuracy–Compute trade-off chart
    # plot_accuracy_compute_tradeoff()
    
    # Plot Accuracy–Latency trade-off chart
    plot_accuracy_latency_tradeoff()