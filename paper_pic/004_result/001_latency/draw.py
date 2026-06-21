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
        - "lat": np.array shape (2, 3):
          rows [w4a4g4, w8a8g8]
          cols [Dense, BDS-unfused, BDS-fused]
        - "ref": float:
          Dense w1a1 latency (optional; can be None to hide)
        - "err": np.array shape (2, 3) or None (optional)

    Speed ratio normalized to w4 Dense:
      baseline = latency(w4 Dense)
      speed(x) = baseline / latency(x)

    Therefore:
      speed(w4 Dense) = 1

    If ref exists, it is plotted as the seventh point:
      speed(ref) = baseline / ref
    """

    if len(sizes) != 4:
        raise ValueError(
            "This 2x2 layout requires exactly four matrix sizes."
        )

    labels_groups = ["w4a4  g=4", "w8a8  g=8"]
    methods = ["Dense", "BDS-unfused", "BDS-fused"]

    # Okabe-Ito-inspired colorblind-friendly colors
    c_dense = "#ABBD4A"
    c_unfuse = "#BDD133"
    c_fuse = "#3B9570"
    c_ref = "#7F7F7F"
    colors = [c_dense, c_unfuse, c_fuse]

    # Speed-ratio line style
    c_speed = "#6F4C9B"
    speed_marker = "o"
    speed_lw = 1.4
    speed_ms = 3.2

    # Bar geometry
    bar_w = 0.26
    inner_gap = 0.00
    offsets = (np.arange(3) - 1) * (bar_w + inner_gap)

    # X positions for two groups and one reference bar
    group_x = np.array([0.0, 1.05])
    ref_x = 1.85

    # --------------------- Figure: 2 x 2 layout ---------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.8, 6.8),
        sharey=share_y,
    )

    # Flatten the 2-D axes array so axes[idx] can still be used
    axes = axes.flatten()

    # Determine global latency y-limit
    global_max = 0.0

    for s in sizes:
        lat = np.asarray(latency_data[s]["lat"], dtype=float)
        global_max = max(global_max, float(np.max(lat)))

        ref = latency_data[s].get("ref", None)
        if ref is not None:
            global_max = max(global_max, float(ref))

    y_lim_top = global_max * 1.25

    # Determine global speed-ratio y-limit
    global_speed_max = 1.0

    for s in sizes:
        lat = np.asarray(latency_data[s]["lat"], dtype=float)

        # w4 Dense baseline
        base = float(lat[0, 0])
        speed = base / lat

        global_speed_max = max(
            global_speed_max,
            float(np.max(speed)),
        )

        ref = latency_data[s].get("ref", None)
        if ref is not None:
            global_speed_max = max(
                global_speed_max,
                float(base / float(ref)),
            )

    speed_lim_top = global_speed_max * 1.25

    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    for idx, s in enumerate(sizes):
        ax = axes[idx]

        lat = np.asarray(
            latency_data[s]["lat"],
            dtype=float,
        )

        err = latency_data[s].get("err", None)
        if err is not None:
            err = np.asarray(err, dtype=float)

        # --------------------- Latency bars ---------------------
        # Two precision/group configurations, each with three bars
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
                label=methods[j] if idx == 0 else None,
            )

        # Dense w1a1 reference bar
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

        # --------------------- Speedup annotations ---------------------
        # Speedup of BDS-unfused and BDS-fused over Dense
        # within the same precision/group configuration
        for gi in range(2):
            dense = lat[gi, 0]

            for j in [1, 2]:
                speedup = dense / lat[gi, j]

                x = group_x[gi] + offsets[j]
                y = lat[gi, j]

                ax.text(
                    x,
                    y + 0.03 * y_lim_top,
                    f"{speedup:.2f}×",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                )

        # --------------------- Speed-ratio line ---------------------
        # Normalize all points to w4a4 Dense
        base = float(lat[0, 0])
        speed = base / lat

        x_pts = np.concatenate([
            group_x[0] + offsets,
            group_x[1] + offsets,
        ])

        y_pts = np.concatenate([
            speed[0, :],
            speed[1, :],
        ])

        if ref is not None:
            x_pts = np.concatenate([
                x_pts,
                np.array([ref_x]),
            ])

            y_pts = np.concatenate([
                y_pts,
                np.array([base / float(ref)]),
            ])

        axr = ax.twinx()

        axr.plot(
            x_pts,
            y_pts,
            color=c_speed,
            linewidth=speed_lw,
            marker=speed_marker,
            markersize=speed_ms,
            zorder=4,
        )

        axr.set_ylim(0, speed_lim_top)

        # Show speed-ratio tick labels only in the right column
        if idx % 2 == 1:
            axr.tick_params(
                axis="y",
                which="both",
                labelright=True,
                right=True,
            )
        else:
            axr.tick_params(
                axis="y",
                which="both",
                labelright=False,
                right=True,
            )

        # Put the speed-ratio axis title on both right-column panels
        if idx % 2 == 1:
            axr.set_ylabel("Speed ratio", fontsize=12)
        else:
            axr.set_ylabel("")

        axr.spines["top"].set_visible(False)
        axr.spines["left"].set_visible(False)

        # --------------------- X axis ---------------------
        if ref is not None:
            ax.set_xticks([
                group_x[0],
                group_x[1],
                ref_x,
            ])

            ax.set_xticklabels(
                labels_groups + ["w1a1"],
                fontsize=10.5,
            )
        else:
            ax.set_xticks([
                group_x[0],
                group_x[1],
            ])

            ax.set_xticklabels(
                labels_groups,
                fontsize=10.5,
            )

        # --------------------- Left Y axis ---------------------
        if share_y:
            ax.set_ylim(0, y_lim_top)
        else:
            local_max = float(np.max(lat))

            if ref is not None:
                local_max = max(local_max, float(ref))

            ax.set_ylim(0, local_max * 1.25)

        # Put latency label on both left-column panels
        if idx % 2 == 0:
            ax.set_ylabel("Latency (ms)", fontsize=12)

        # Grid and spines
        ax.grid(
            axis="y",
            linestyle="-",
            linewidth=0.6,
            alpha=0.22,
            zorder=0,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel tag
        ax.text(
            0.02,
            0.96,
            panel_tags[idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

        # Matrix-size title
        ax.set_title(
            f"{s}x{s}x{s}",
            fontsize=12,
            pad=6,
        )

        # Separator before the w1a1 reference bar
        if ref is not None:
            separator_x = (
                group_x[1] + bar_w + ref_x
            ) / 2

            ax.axvline(
                x=separator_x,
                color="black",
                linewidth=0.8,
                alpha=0.18,
            )

    # --------------------- Global legend ---------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fancybox=False,
        edgecolor="black",
        fontsize=10,
        bbox_to_anchor=(0.5, 0.995),
    )

    # --------------------- Layout ---------------------
    fig.subplots_adjust(
        left=0.10,
        right=0.91,
        top=0.88,
        bottom=0.10,
        wspace=0.12,
        hspace=0.32,
    )

    # Save outputs
    fig.savefig(
        f"{save_prefix}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        f"{save_prefix}.pdf",
        bbox_inches="tight",
    )

    return fig, axes


if __name__ == "__main__":
    # lat:
    # [[w4a4g4 Dense, BDS-unfused, BDS-fused],
    #  [w8a8g8 Dense, BDS-unfused, BDS-fused]]

    latency_data = {
        128: {
            "lat": np.array([
                [0.006031, 0.010687, 0.004195],
                [0.012460, 0.010528, 0.004902],
            ]),
            "ref": 0.006528,
        },

        256: {
            "lat": np.array([
                [0.008341, 0.011210, 0.004664],
                [0.012580, 0.011732, 0.004943],
            ]),
            "ref": 0.007640,
        },

        512: {
            "lat": np.array([
                [0.010104, 0.011523, 0.004849],
                [0.015627, 0.011741, 0.005387],
            ]),
            "ref": 0.008324,
        },

        1024: {
            "lat": np.array([
                [0.014376, 0.012480, 0.006280],
                [0.020780, 0.012560, 0.006860],
            ]),
            "ref": 0.009695,
        },
    }

    draw_latency_multi_sizes(
        latency_data=latency_data,
        sizes=(128, 256, 512, 1024),
        save_prefix="latency_multi_sizes_2x2",
        share_y=True,
    )

    plt.show()