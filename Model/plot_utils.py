import numpy as np
import os
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba, LinearSegmentedColormap
import networkx as nx



def visualize_matrices(
    matrices: dict,
    *,
    y_true=None,
    cmap_labels: str = "Set3",
    figsize_per_panel=(4.5, 5.0),
    save_path: str | None = None,
):
    """
    Visualize multiple matrices from a dict: {title: matrix}.
    - optional y_true will sort all matrices (same order) and show a label bar
    """
    # ---------- sorting (optional) ----------
    y_sorted = None
    if y_true is not None:
        y_true = y_true.astype(int).reshape(-1)
        n = next(iter(matrices.values())).shape[0]
        if y_true.shape[0] != n:
            raise ValueError(f"y_true length {y_true.shape[0]} must match matrix size {n}")
        order = np.argsort(y_true)
        y_sorted = y_true[order]
        matrices = {k: v[order][:, order] for k, v in matrices.items()}

    # ---------- colormap (white at 0) ----------
    base = plt.cm.get_cmap("GnBu", 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  # RGBA white
    cmap_matrix = LinearSegmentedColormap.from_list("white0_gnbu", colors)

    titles = list(matrices.keys())
    n_panels = len(titles)
    has_labels = y_sorted is not None

    fig_w = (0.6 if has_labels else 0.0) + figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        2 if has_labels else 1,
        (1 if has_labels else 0) + n_panels,
        width_ratios=([0.05] if has_labels else []) + [1] * n_panels,
        height_ratios=[0.05, 1] if has_labels else [1],
        wspace=0.05,
        hspace=0.05,
    )
    col_offset = 1 if has_labels else 0

    # ---------- label bar ----------
    if has_labels:
        labels = np.unique(y_sorted)
        num_classes = len(labels)
        base_cmap = plt.cm.get_cmap(cmap_labels)
        label_colors = [base_cmap(i) for i in range(num_classes)]
        label_cmap = ListedColormap(label_colors)
        bounds = np.arange(num_classes + 1) - 0.5
        norm = BoundaryNorm(bounds, label_cmap.N)

        ax_label = fig.add_subplot(gs[-1, 0])
        ax_label.imshow(y_sorted[:, None], aspect="auto", cmap=label_cmap, norm=norm)
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        ax_label.set_ylabel("y_true", rotation=90, labelpad=10)

    # ---------- heatmaps ----------
    axes = []
    last_im = None
    for i, title in enumerate(titles):
        ax = fig.add_subplot(gs[-1, i + col_offset])
        hm = sns.heatmap(
            matrices[title],
            cmap=cmap_matrix,
            square=True,
            cbar=False,
            ax=ax,
        )
        ax.set_title(str(title))
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        last_im = hm

    # shared colorbar
    cbar = fig.colorbar(last_im.collections[0], ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Weight", rotation=270, labelpad=15)
    plt.savefig(save_path, bbox_inches="tight")



def plot_training_history(history: Dict[str, list], save_path=None):
    epochs = np.arange(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["mean_w"])
    plt.xlabel("Epoch")
    plt.ylabel("mean_w")
    plt.title("Mean edge weight")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path+'plot_training_history.svg')
    plt.show()


def SCI_COLORS():
    res = [
        "#0072B2",  # Blue
        "#E69F00",  # Orange
        "#009E73",  # Teal (Blue-Green)
        "#D55E00",  # Vermilion (Red-Orange)
        "#AA4499",  # Purple
        "#56B4E9",  # Sky Blue
        "#D3C617",  # Yellow (Golden Yellow)
        "#117733",  # Dark Green
        "#937860",  # Brown (Warm Brown)
        "#8C8C1D",  # Olive (Olive Yellow)
    ]
    return res


def plot_downstream_clustering_bars(results, save_path=None):
    """
    Plot downstream clustering results in two separate figures:
      1) ARI (Noisy vs Denoised)
      2) NMI (Noisy vs Denoised)

    results: list of dict, each dict like:
      {
        "algo": str,
        "ari_noisy": float or None,
        "ari_den": float or None,
        "nmi_noisy": float or None,
        "nmi_den": float or None,
      }
    """
    def clean_algo_name(name: str) -> str:
        # Remove package prefix like "igraph_" / "cdlib_" and keep the rest
        if "_" not in name:
            return name
        return name.split("_", 1)[1]

    def light(color, alpha=0.35):
        r, g, b, _ = to_rgba(color)
        return r, g, b, alpha

    # Filter out completely invalid entries
    results = [
        r for r in results
        if (r.get("ari_noisy") is not None or r.get("ari_den") is not None or
            r.get("nmi_noisy") is not None or r.get("nmi_den") is not None)
    ]

    algos = [clean_algo_name(r["algo"]) for r in results]
    n = len(algos)

    x = np.arange(n)
    width = 0.35

    colors = SCI_COLORS()[0:n]

    # =======================
    # Figure 1: ARI
    # =======================
    fig1, ax1 = plt.subplots(figsize=(max(10, round(n * 0.55)), 4))

    for i, r in enumerate(results):
        if r.get("ari_noisy") is not None:
            ax1.bar(
                x[i] - width / 2,
                r["ari_noisy"],
                width,
                color=light(colors[i]),
            )
        if r.get("ari_den") is not None:
            ax1.bar(
                x[i] + width / 2,
                r["ari_den"],
                width,
                color=colors[i],
            )

    ax1.set_title("ARI (Noisy vs Denoised)")
    ax1.set_ylabel("ARI")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path + 'ARI_Noisy_vs_Denoised.svg')
    plt.show()

    # =======================
    # Figure 2: NMI
    # =======================
    fig2, ax2 = plt.subplots(figsize=(max(10, round(n * 0.55)), 4))

    for i, r in enumerate(results):
        if r.get("nmi_noisy") is not None:
            ax2.bar(
                x[i] - width / 2,
                r["nmi_noisy"],
                width,
                color=light(colors[i]),
            )
        if r.get("nmi_den") is not None:
            ax2.bar(
                x[i] + width / 2,
                r["nmi_den"],
                width,
                color=colors[i],
            )

    ax2.set_title("NMI (Noisy vs Denoised)")
    ax2.set_ylabel("NMI")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path + 'NMI_Noisy_vs_Denoised.svg')
    plt.show()



def visualize_network_dict(
    graphs_dict: dict,
    y_true: np.ndarray,
    min_weight_to_draw: float = 0.0,
    layout: str = 'spring',
    save_dir: str = None,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    method_names = list(graphs_dict.keys())
    N = next(iter(graphs_dict.values())).shape[0]

    # ----- node colors -----
    num_classes = len(np.unique(y_true))
    cmap = plt.cm.get_cmap("tab10", num_classes)
    node_colors = [cmap(int(y_true[i])) for i in range(N)]

    # ----- 用第一个图固定 layout（保证可比）-----
    A_layout = next(iter(graphs_dict.values()))
    G_layout = nx.from_numpy_array((A_layout > 0).astype(int))

    if layout == 'spring':
        pos = nx.spring_layout(G_layout, seed=64)
    else:
        pos, _ = community_aware_layout(y_true, G_layout, radius=2.0, spread=3.3)

    # =========================================================
    # 每个方法单独画
    # =========================================================
    for name in method_names:
        A = graphs_dict[name]

        G = nx.Graph()
        G.add_nodes_from(range(N))

        iu = np.triu_indices(N, k=1)
        ii, jj = iu
        ww = A[iu]

        mask = np.isfinite(ww) & (ww > min_weight_to_draw)

        for u, v, w in zip(ii[mask], jj[mask], ww[mask]):
            G.add_edge(int(u), int(v), weight=float(w))

        # ---- 分 intra / inter ----
        intra_edges, inter_edges = [], []
        for u, v in G.edges():
            if y_true[u] == y_true[v]:
                intra_edges.append((u, v))
            else:
                inter_edges.append((u, v))

        # =====================================================
        # Plot
        # =====================================================
        plt.figure(figsize=(5, 4))

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=300,
            edgecolors="k",
            linewidths=0.5
        )

        def edge_width(u, v):
            w = float(G[u][v].get("weight", 0.0))
            return 0.5 + 3.0 * w

        if intra_edges:
            widths = [edge_width(u, v) for (u, v) in intra_edges]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=intra_edges,
                edge_color="grey",
                width=widths,
                alpha=0.8
            )

        if inter_edges:
            widths = [edge_width(u, v) for (u, v) in inter_edges]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=inter_edges,
                edge_color="red",
                width=widths,
                alpha=0.5
            )

        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f"{name} (|E|={G.number_of_edges()})")
        plt.axis("off")

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{name}.svg"))

        plt.show()


def community_aware_layout(
        y_np,
        G=None,          # Optional graph, used for spring layout inside each community
        seed=0,
        radius=6.0,      # Distance between communities (larger -> more separated)
        spread=2.5,      # Intra-community dispersion (larger -> looser clusters)
):
    rng = np.random.default_rng(seed)

    labels = np.unique(y_np)
    K = len(labels)

    # ===== Place community centers evenly on a circle =====
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    centers = {
        lab: np.array([radius*np.cos(a), radius*np.sin(a)])
        for lab, a in zip(labels, angles)
    }

    pos = {}

    for lab in labels:
        nodes = np.where(y_np == lab)[0]

        # ----- Local layout inside each community -----
        if G is not None:
            # Spring layout constrained to the subgraph
            subG = G.subgraph(nodes)
            local_pos = nx.spring_layout(
                subG,
                seed=seed,
                k=spread / np.sqrt(len(nodes)),  # controls node spacing
                iterations=80
            )
        else:
            # Random Gaussian layout if no graph structure is available
            local_pos = {
                i: rng.normal(scale=spread, size=2)
                for i in nodes
            }

        # ----- Shift local coordinates to the community center -----
        for i in nodes:
            pos[int(i)] = centers[lab] + np.array(local_pos[i])

    return pos, centers



def get_method_color_map(methods):
    base_colors = {
        "Noisy": "#B0B0B0",
        "CLEAN": "#0072B2",
        "NR": "#E69F00",
        "CN": "#009E73",
        "Katz": "#D55E00",
        "LP": "#AA4499",
        "NE": "#117733",
    }

    fallback = [
        "#117733", "#CC79A7", "#999999"
    ]
    color_map = {}
    i = 0
    for m in methods:
        if m in base_colors:
            color_map[m] = base_colors[m]
        else:
            color_map[m] = fallback[i % len(fallback)]
            i += 1

    return color_map

def plot_snr_bench(df, save_path):
    methods_all = df["method"].tolist()
    method_order = sorted(
        methods_all,
        key=lambda m: df.loc[df["method"] == m, "community.snr"].values[0]
    )

    df_plot = df.set_index("method").loc[method_order].reset_index()
    color_map = get_method_color_map(method_order)

    methods = df_plot["method"].values
    scores = df_plot["community.snr"].values
    colors = [color_map[m] for m in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, scores, color=colors, width=0.6)

    plt.ylabel("SNR score")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "snr_barplot.svg"))
    plt.show()

def extract_clustering_scores_bench(df, metric="ari"):
    results = {}

    for _, row in df.iterrows():
        method = row["method"]
        vals = []

        for col in df.columns:
            if f".{metric}" in col and col.startswith("clustering."):
                val = row[col]
                if not np.isnan(val):
                    vals.append(val)

        results[method] = vals

    return results


# =========================================================
# ARI / NMI boxplot + scatter
# =========================================================
def plot_clustering_box_bench(df, metric, save_path):
    data_dict = extract_clustering_scores_bench(df, metric)

    methods_all = list(data_dict.keys())
    method_order = sorted(
        methods_all,
        key=lambda m: np.mean(data_dict[m]) if len(data_dict[m]) > 0 else -np.inf
    )

    color_map = get_method_color_map(method_order)

    methods = method_order
    data = [data_dict[m] for m in methods]

    plt.figure(figsize=(8, 5))

    box = plt.boxplot(
        data,
        patch_artist=True,
        labels=methods,
        showfliers=False
    )

    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(color_map[method])
        patch.set_alpha(0.6)

    # scatter
    for i, method in enumerate(methods):
        vals = data_dict[method]
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        plt.scatter(
            x,
            vals,
            color=color_map[method],
            edgecolors="black",
            linewidths=0.5,
            s=35,
            alpha=0.9,
            zorder=3
        )

    plt.xticks(rotation=30)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} Distribution Across Clustering Algorithms")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric}_boxplot.svg"))
    plt.show()