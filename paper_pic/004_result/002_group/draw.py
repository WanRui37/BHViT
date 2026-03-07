import numpy as np
import matplotlib.pyplot as plt
import re

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

# ====== 1) Paste your printed groups here ======
RAW = r"""
Current Groups:
  module.blocks.0.attn.qkv: 3
  module.blocks.0.attn.proj: 2
  module.blocks.0.mlp.fc1: 1
  module.blocks.0.mlp.fc2: 1
  module.blocks.1.attn.qkv: 8
  module.blocks.1.attn.proj: 8
  module.blocks.1.mlp.fc1: 12
  module.blocks.1.mlp.fc2: 8
  module.blocks.2.attn.qkv: 2
  module.blocks.2.attn.proj: 2
  module.blocks.2.mlp.fc1: 12
  module.blocks.2.mlp.fc2: 8
  module.blocks.3.attn.qkv: 2
  module.blocks.3.attn.proj: 2
  module.blocks.3.mlp.fc1: 12
  module.blocks.3.mlp.fc2: 12
  module.blocks.4.attn.qkv: 2
  module.blocks.4.attn.proj: 2
  module.blocks.4.mlp.fc1: 12
  module.blocks.4.mlp.fc2: 12
  module.blocks.5.attn.qkv: 2
  module.blocks.5.attn.proj: 2
  module.blocks.5.mlp.fc1: 12
  module.blocks.5.mlp.fc2: 12
  module.blocks.6.attn.qkv: 8
  module.blocks.6.attn.proj: 12
  module.blocks.6.mlp.fc1: 12
  module.blocks.6.mlp.fc2: 12
  module.blocks.7.attn.qkv: 8
  module.blocks.7.attn.proj: 12
  module.blocks.7.mlp.fc1: 12
  module.blocks.7.mlp.fc2: 12
  module.blocks.8.attn.qkv: 2
  module.blocks.8.attn.proj: 3
  module.blocks.8.mlp.fc1: 12
  module.blocks.8.mlp.fc2: 12
  module.blocks.9.attn.qkv: 8
  module.blocks.9.attn.proj: 12
  module.blocks.9.mlp.fc1: 12
  module.blocks.9.mlp.fc2: 12
  module.blocks.10.attn.qkv: 8
  module.blocks.10.attn.proj: 12
  module.blocks.10.mlp.fc1: 8
  module.blocks.10.mlp.fc2: 4
  module.blocks.11.attn.qkv: 12
  module.blocks.11.attn.proj: 12
  module.blocks.11.mlp.fc1: 1
  module.blocks.11.mlp.fc2: 2
"""

# ====== 2) Parse (block_idx, proj_type) -> G ======
# We normalize projection names into 4 categories:
#   attn.qkv, attn.proj, mlp.fc1, mlp.fc2
pattern = re.compile(
    r"module\.blocks\.(\d+)\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2):\s*(\d+)"
)

groups = {}  # (proj, layer) -> G
max_layer = -1
for line in RAW.splitlines():
    m = pattern.search(line.strip())
    if not m:
        continue
    layer = int(m.group(1))
    proj = m.group(2)
    g = int(m.group(3))
    groups[(proj, layer)] = g
    max_layer = max(max_layer, layer)

L = max_layer + 1
projs = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

# Build matrix: rows=proj, cols=layer
M = np.full((len(projs), L), np.nan, dtype=float)
for i, proj in enumerate(projs):
    for layer in range(L):
        if (proj, layer) in groups:
            M[i, layer] = groups[(proj, layer)]

# Sanity check: warn if any missing
missing = [(proj, layer) for proj in projs for layer in range(L) if np.isnan(M[projs.index(proj), layer])]
if missing:
    print("WARNING: missing entries:", missing)

# ====== 3) Figure A: layer-wise average G (searched) vs fixed G=4 ======
avg_search = np.nanmean(M, axis=0)
avg_fixed = np.full(L, 4.0)

plt.figure(figsize=(7.8, 2.8))
xs = np.arange(L)
plt.plot(xs, avg_fixed, marker="o", linewidth=1.5, label="Fixed (G=4)")
plt.plot(xs, avg_search, marker="o", linewidth=1.5, label="Searched (avg over projections)")
plt.xticks(xs, [str(i) for i in xs])
plt.xlabel("Block index")
plt.ylabel("Group number G")
plt.legend()
plt.tight_layout()
plt.savefig("groups_layerwise_avg.pdf", bbox_inches="tight")
plt.savefig("groups_layerwise_avg.png", dpi=300, bbox_inches="tight")
plt.show()

# ====== 4) Figure B: heatmap (proj x layer) ======
plt.figure(figsize=(8.4, 2.6))
im = plt.imshow(M, aspect="auto", interpolation="nearest")
plt.yticks(np.arange(len(projs)), projs)
plt.xticks(np.arange(L), [str(i) for i in range(L)])
plt.xlabel("Block index")
cbar = plt.colorbar(im)
cbar.set_label("Group number G")
plt.tight_layout()
plt.savefig("groups_heatmap.pdf", bbox_inches="tight")
plt.savefig("groups_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# ====== 5) (Optional) Figure C: grouped bars per layer ======
# This is a bit wider; useful if you want a very explicit view.
plt.figure(figsize=(9.6, 3.0))
width = 0.18
offsets = np.linspace(-1.5*width, 1.5*width, len(projs))
for i, proj in enumerate(projs):
    plt.bar(xs + offsets[i], M[i], width=width, label=proj)
plt.axhline(4.0, linestyle="--", linewidth=1.0, label="Fixed (G=4)")
plt.plot(xs, avg_search, marker="o", linewidth=1.5, color="darkblue", label="Searched (avg)")
plt.xticks(xs, [str(i) for i in xs])
plt.xlabel("Block index")
plt.ylabel("Group number G")
plt.legend(ncol=4, fontsize=9)
plt.tight_layout()
plt.savefig("groups_bars.pdf", bbox_inches="tight")
plt.savefig("groups_bars.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: groups_layerwise_avg.(pdf/png), groups_heatmap.(pdf/png), groups_bars.(pdf/png)")