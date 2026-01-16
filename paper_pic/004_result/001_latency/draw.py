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

