import numpy as np
import matplotlib.pyplot as plt

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
def draw_speedup(latency_data):

    sizes = list(latency_data.keys())
    labels = [f"{s}/{s}/{s}" if s < 1000 else f"{int(s/1000)}k/{int(s/1000)}k/{int(s/1000)}k" for s in sizes]

    int8 = []
    int4 = []
    int1 = []

    for s in sizes:
        ref = latency_data[s]["ref"]
        lat = latency_data[s]["lat"]

        # choose representative latency
        lat_int8 = lat[1,0]  # w8a8 Dense
        lat_int4 = lat[0,0]  # w4a4 Dense
        lat_int1 = ref  # fastest fused

        int8.append(lat_int4/lat_int8)
        int4.append(lat_int4/lat_int4)
        int1.append(lat_int4/lat_int1)

    int8 = np.array(int8)
    int4 = np.array(int4)
    int1 = np.array(int1)

    x = np.arange(len(sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(6,6))

    colors = ["#C3D63C","#4DA37A","#0B5D5E"]

    ax.bar(x-width, int8, width, label="INT8", color=colors[0])
    ax.bar(x, int4, width, label="INT4", color=colors[1])
    ax.bar(x+width, int1, width, label="INT1", color=colors[2])

    ax.set_ylabel("Speed ratio")
    ax.set_xlabel("Matrix size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, max(int1)*1.05)

    # 添加水平网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.5)

    ax.legend(loc="upper left", ncol=3, frameon=False)

    plt.tight_layout()

    plt.savefig("speedup_matrix_sizes.png", dpi=300, bbox_inches="tight")
    plt.savefig("speedup_matrix_sizes.pdf", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":

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

    draw_speedup(latency_data)