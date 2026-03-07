import numpy as np
import matplotlib.pyplot as plt

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

def draw_latency_multi_sizes(
    latency_data,
    sizes=(128, 256, 512, 1024),
    save_prefix="latency_multi_sizes",
    share_y=True,
):
    """
    latency_data: dict
      key: size (int), value: dict with keys:
        - "lat": np.array shape (2,3): rows [w4a4g4, w8a8g8], cols [Dense, BDS-unfused, BDS-fused]
        - "ref": float: Dense w1a1 latency (optional; can be None to hide)
        - "err": np.array shape (2,3) or None (optional)
    """

    labels_groups = ["w4a4  g=4", "w8a8  g=8"]
    methods = ["Dense", "BDS-unfused", "BDS-fused"]

    # Okabe-Ito (colorblind-friendly)
    c_dense  = "#ABBD4A"  # blue
    c_unfuse = "#BDD133"  # orange
    c_fuse   = "#3B9570"  # green
    c_ref    = "#7F7F7F"  # gray
    colors = [c_dense, c_unfuse, c_fuse]

    # Tighter bar spacing (you said spacing too large)
    bar_w = 0.26
    inner_gap = 0.00  # smaller -> bars closer
    offsets = (np.arange(3) - 1) * (bar_w + inner_gap)

    # x positions for 2 groups + ref, keep compact
    group_x = np.array([0.0, 1.05])  # bring the two groups closer
    ref_x = 1.85                     # bring reference closer too

    # Figure: 4 subplots in one row
    fig, axes = plt.subplots(
        1, len(sizes),
        figsize=(15.8, 3.8),
        sharey=share_y
    )

    # Determine global ymax for shared y-axis (paper-friendly)
    global_max = 0.0
    for s in sizes:
        lat = np.asarray(latency_data[s]["lat"], dtype=float)
        global_max = max(global_max, float(np.max(lat)))
        ref = latency_data[s].get("ref", None)
        if ref is not None:
            global_max = max(global_max, float(ref))
    y_lim_top = global_max * 1.25

    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    for idx, s in enumerate(sizes):
        ax = axes[idx]

        lat = np.asarray(latency_data[s]["lat"], dtype=float)   # (2,3)
        err = latency_data[s].get("err", None)
        if err is not None:
            err = np.asarray(err, dtype=float)

        # Draw 2 groups, each has 3 bars
        for j in range(3):
            ax.bar(
                group_x + offsets[j],
                lat[:, j],
                width=bar_w,
                color=colors[j],
                edgecolor="black",
                linewidth=0,
                yerr=None if err is None else err[:, j],
                capsize=2.5 if err is not None else 0,
                zorder=3,
                label=methods[j] if idx == 0 else None,  # only add legend once
            )

        # Reference bar
        ref = latency_data[s].get("ref", None)
        if ref is not None:
            ax.bar(
                ref_x,
                ref,
                width=bar_w,
                color=c_ref,
                edgecolor="black",
                linewidth=0,
                alpha=0.75,
                hatch="///",
                zorder=3,
                label="Dense w1a1 (ref.)" if idx == 0 else None,
            )

        # Speedup annotations (vs Dense within each group)
        for gi in range(2):  # 0: w4a4g4, 1: w8a8g8
            dense = lat[gi, 0]
            # annotate above unfused and fused
            for j in [1, 2]:
                sp = dense / lat[gi, j]
                x = group_x[gi] + offsets[j]
                y = lat[gi, j]
                ax.text(
                    x, y + 0.03 * y_lim_top,
                    f"{sp:.2f}×",
                    ha="center", va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                )

        # X ticks: group centers + ref
        ax.set_xticks([group_x[0], group_x[1], ref_x] if ref is not None else [group_x[0], group_x[1]])
        ax.set_xticklabels(labels_groups + (["ref."] if ref is not None else []), fontsize=11)

        # Y axis
        ax.set_ylim(0, y_lim_top if share_y else max(np.max(lat), (ref or 0)) * 1.25)
        if idx == 0:
            ax.set_ylabel("Latency (ms)", fontsize=12)

        # Grid + clean spines
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.22, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel tag and title (size)
        ax.text(0.02, 0.96, panel_tags[idx], transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")
        ax.set_title(f"{s}x{s}x{s}", fontsize=12, pad=6)

        # Optional subtle separator before ref
        if ref is not None:
            ax.axvline(x=(group_x[1] + bar_w + ref_x) / 2, color="black", linewidth=0.8, alpha=0.18)

    # Legend (single, global, top center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fancybox=False,
        edgecolor="black",
        fontsize=10,
        bbox_to_anchor=(0.5, 1.0)
    )

    # Tight layout: reduce gaps between subplots
    fig.subplots_adjust(left=0.06, right=0.995, top=0.78, bottom=0.18, wspace=0.05)

    fig.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    # ------------------ Example placeholder data ------------------
    # Replace these with your real measurements.
    # lat: [[w4a4g4 Dense, unfused, fused],
    #       [w8a8g8 Dense, unfused, fused]]
    latency_data = {
        128: {
            "lat": np.array([[0.006031, 0.010687, 0.004195],
                             [0.012460, 0.010528, 0.004902]]),
            "ref": 0.006528,
        },
        256: {
            "lat": np.array([[0.008341, 0.011210, 0.004664],
                             [0.012580, 0.011732, 0.004943]]),
            "ref": 0.00764,
        },
        512: {
            "lat": np.array([[0.010104, 0.011523, 0.004849],
                             [0.015627, 0.011741, 0.005087]]),
            "ref": 0.008324,
        },
        1024: {
            "lat": np.array([[0.014376, 0.012480, 0.005280],
                             [0.020780, 0.012560, 0.005460]]),
            "ref": 0.009695,
        },
    }

    draw_latency_multi_sizes(latency_data)