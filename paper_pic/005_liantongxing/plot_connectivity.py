import matplotlib.pyplot as plt
import numpy as np

# --------------------- Global style (paper-ready) ---------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
})

# Data generation
depths = np.arange(1, 13)

# Define offsets for visualization
offset_base = 0.1
offsets = [0, 0.05, 0.1, 0.15] # For 'no residual' curves to show small vertical offsets
residual_offset_1 = 0.1 # Small offset for residual curves
residual_offset_2 = 0.2 # Another offset for residual curves

# Store data for plotting
plot_data = []

# Curve: shift=0 (residual)
y_shift0_res = np.array([1.0] * len(depths))
plot_data.append({'x': depths, 'y': y_shift0_res, 'label': 'shift=0 (residual)', 'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'})

# Curve: shift=0 (no residual)
y_shift0_no_res = np.array([1.0 - offsets[0]] * len(depths))
plot_data.append({'x': depths, 'y': y_shift0_no_res, 'label': 'shift=0 (no residual)', 'color': '#1f77b4', 'linestyle': '--', 'marker': 's'})

# Curve: shift=1 (residual)
y_shift1_res = np.array([min(d + 1, 8) - residual_offset_1 for d in depths])
plot_data.append({'x': depths, 'y': y_shift1_res, 'label': 'shift=1 (residual)', 'color': '#bcbd22', 'linestyle': '-', 'marker': 'o'})

# Curve: shift=1 (no residual)
y_shift1_no_res = np.array([1.0 - offsets[1]] * len(depths))
plot_data.append({'x': depths, 'y': y_shift1_no_res, 'label': 'shift=1 (no residual)', 'color': '#bcbd22', 'linestyle': '--', 'marker': 's'})

# Curve: shift=2 (residual)
y_shift2_res = np.array([min(2 + (d - 1) * (2/3), 4) for d in depths])
plot_data.append({'x': depths, 'y': y_shift2_res, 'label': 'shift=2 (residual)', 'color': '#ff7f0e', 'linestyle': '-', 'marker': 'o'})

# Curve: shift=2 (no residual)
y_shift2_no_res = np.array([1.0 - offsets[2]] * len(depths))
plot_data.append({'x': depths, 'y': y_shift2_no_res, 'label': 'shift=2 (no residual)', 'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'})

# Curve: shift=3 (residual) - same as shift=1 residual
y_shift3_res = np.array([min(d + 1, 8) - residual_offset_2 for d in depths])
plot_data.append({'x': depths, 'y': y_shift3_res, 'label': 'shift=3 (residual)', 'color': '#2ca02c', 'linestyle': '-', 'marker': 'o'})

# Curve: shift=3 (no residual)
y_shift3_no_res = np.array([1.0 - offsets[3]] * len(depths))
plot_data.append({'x': depths, 'y': y_shift3_no_res, 'label': 'shift=3 (no residual)', 'color': '#2ca02c', 'linestyle': '--', 'marker': 's'})


# Plotting
plt.figure(figsize=(10, 4))

for data in plot_data:
    plt.plot(data['x'], data['y'], label=data['label'], color=data['color'],
             linestyle=data['linestyle'], marker=data['marker'], markersize=5, linewidth=2)

plt.xlabel('Depth', fontsize=12)
plt.ylabel('Reachable groups', fontsize=12)
plt.xticks(np.arange(0, 13, 2))
plt.yticks(np.arange(0, 9, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(ncol=2, loc='upper left', frameon=False, fontsize=10)
plt.xlim(0.5, 12.5)
plt.ylim(0.0, 8.5)

plt.tight_layout()

# Save the figure
plt.savefig('/mnt/wr/3-LLM/9-ViT/BHViT/paper_pic/005_liantongxing/figure6.pdf')
plt.close()

print("Plot generated successfully at /mnt/wr/3-LLM/9-ViT/BHViT/paper_pic/005_liantongxing/figure6.pdf")
