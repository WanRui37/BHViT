import matplotlib.pyplot as plt
import numpy as np
import random

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
num_layers = 12
depths = np.arange(1, num_layers + 1)

# Define possible group numbers
possible_group_numbers = [2, 4, 6, 8, 12]

# Randomly assign a group number for each layer
# Using a fixed seed for reproducibility for now
random.seed(42)
layer_group_assignments = [random.choice(possible_group_numbers) for _ in range(num_layers)]

print(f"Layer group assignments: {layer_group_assignments}")

# Store data for plotting
plot_data = []

# Colors for different shift values
colors = {
    'shift=0 (residual)': '#1f77b4',
    'shift=0 (no residual)': '#1f77b4',
    'shift=1 (residual)': '#bcbd22',
    'shift=1 (no residual)': '#bcbd22',
    'shift=2 (residual)': '#ff7f0e',
    'shift=2 (no residual)': '#ff7f0e',
    'shift=3 (residual)': '#2ca02c',
    'shift=3 (no residual)': '#2ca02c',
}

# Line styles for residual vs no residual
linestyles = {
    'residual': '-',
    'no residual': '--'
}

markers = {
    'residual': 'o',
    'no residual': 's'
}

# Offsets to distinguish lines for better visualization (trial and error based)
offsets = {
    'shift=0 (residual)': 0.0,
    'shift=0 (no residual)': -0.2,
    'shift=1 (residual)': 0.1,
    'shift=1 (no residual)': -0.1,
    'shift=2 (residual)': 0.2,
    'shift=2 (no residual)': -0.3,
    'shift=3 (residual)': 0.3,
    'shift=3 (no residual)': -0.4,
}

# Define the curves to plot
curve_configs = [
    {'shift': 0, 'residual': True, 'label': 'shift=0 (residual)'},
    {'shift': 0, 'residual': False, 'label': 'shift=0 (no residual)'},
    {'shift': 1, 'residual': True, 'label': 'shift=1 (residual)'},
    {'shift': 1, 'residual': False, 'label': 'shift=1 (no residual)'},
    {'shift': 2, 'residual': True, 'label': 'shift=2 (residual)'},
    {'shift': 2, 'residual': False, 'label': 'shift=2 (no residual)'},
    {'shift': 3, 'residual': True, 'label': 'shift=3 (residual)'},
    {'shift': 3, 'residual': False, 'label': 'shift=3 (no residual)'},
]

for config in curve_configs:
    y_values = []
    current_shift = config['shift']
    has_residual = config['residual']
    label = config['label']
    plot_offset = offsets[label]

    for d_idx, d in enumerate(depths):
        G_for_layer = layer_group_assignments[d_idx] # Get the group number for the current layer

        if not has_residual:
            # For no residual, reachable groups is always 1 (with offset)
            y_values.append(1.0 + plot_offset)
        else:
            if current_shift == 0:
                reachable = 1 # Only the initial group is reachable
            elif current_shift == 1 or current_shift == 3: # Assuming shift 3 behaves like shift 1 for simplicity based on original plot
                reachable = min(d + 1, G_for_layer) # Reachable groups increase linearly with depth, capped by G
            elif current_shift == 2:
                # This is an approximation based on the original plot's curve for shift=2
                # It appears to be roughly (2 + (d-1)*(2/3)), capped by G
                reachable = min(2 + (d - 1) * (2/3), G_for_layer)
            else:
                reachable = 1 # Default or unknown shift
            y_values.append(reachable + plot_offset)

    plot_data.append({
        'x': depths,
        'y': np.array(y_values),
        'label': label,
        'color': colors[label],
        'linestyle': linestyles['residual'] if has_residual else linestyles['no residual'],
        'marker': markers['residual'] if has_residual else markers['no residual']
    })


# Plotting
plt.figure(figsize=(10, 4))

for data in plot_data:
    plt.plot(data['x'], data['y'], label=data['label'], color=data['color'],
             linestyle=data['linestyle'], marker=data['marker'], markersize=5, linewidth=2)

plt.xlabel('Depth', fontsize=12)
plt.ylabel('Reachable groups (with offset for clarity)', fontsize=12)
plt.xticks(np.arange(0, num_layers + 1, 2))
# Max y-limit should consider the max possible group number plus max offset
max_y_val = max(possible_group_numbers) + max(abs(offset) for offset in offsets.values()) + 0.5
plt.yticks(np.arange(0, max_y_val, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(ncol=2, loc='upper left', frameon=False, fontsize=10)
plt.xlim(0.5, num_layers + 0.5)
plt.ylim(0.0, max_y_val)

# Add group number annotations for each layer
annotation_y_pos = max_y_val - 0.5 # Position for annotations
for d_idx, d in enumerate(depths):
    plt.text(d, annotation_y_pos, f'G={layer_group_assignments[d_idx]}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save the figure
plt.savefig('/mnt/wr/3-LLM/9-ViT/BHViT/paper_pic/005_liantongxing/figure_random_group_per_layer.pdf')
plt.close()

print("Plot generated successfully at /mnt/wr/3-LLM/9-ViT/BHViT/paper_pic/005_liantongxing/figure_random_group_per_layer.pdf")
