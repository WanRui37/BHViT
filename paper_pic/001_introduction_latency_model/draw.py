import numpy as np
import matplotlib.pyplot as plt

# --------------------- Global style ---------------------
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --------------------- Data: fill by yourself ---------------------
models = [
    "BHViT-S (W1A1)",
    "Q-ViT-S (W1A1)",
    "Q-ViT-T (W4A4)",
]

latency = {
    "Projection": [3.9, 3.6, 2.6],
    "Attention":  [2.1, 1.1, 0.9],
    "Other":      [0.8, 0.7, 0.6],
}

colors = {
    "Projection": "#4C78A8",
    "Attention":  "#F58518",
    "Other":      "#54A24B",
}

hatches = {
    "Projection": "",
    "Attention":  "//",
    "Other":      "..",
}

# --------------------- Plot ---------------------
# smaller factor means smaller vertical spacing
y = np.arange(len(models)) * 0.38
left = np.zeros(len(models))

fig, ax = plt.subplots(figsize=(7.2, 2.25))

bar_height = 0.28

for name, values in latency.items():
    values = np.array(values)
    ax.barh(
        y,
        values,
        left=left,
        height=bar_height,
        label=name,
        color=colors[name],
        edgecolor="white",
        linewidth=0.7,
        hatch=hatches[name],
    )
    left += values

totals = left
for i, total in enumerate(totals):
    ax.text(
        total + 0.05,
        y[i],
        f"{total:.1f}",
        va="center",
        ha="left",
        fontsize=11,
    )

ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=11)
ax.invert_yaxis()

ax.set_xlabel("Latency (ms/img)")
ax.set_xlim(0, max(totals) * 1.15)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=3,
    frameon=False,
    fontsize=10,
    handlelength=1.5,
    columnspacing=1.0,
)

ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# reduce blank margins around bars
ax.margins(y=0.08)

plt.tight_layout(pad=0.3)

save_path = "latency_breakdown.pdf"
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
plt.show()

print(f"Saved to: {save_path}")