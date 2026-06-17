import os
import pandas as pd
from Model.evaluation import evaluate_single_network_clustering, snr_cohen_d
from Model.plot_utils import visualize_matrices, visualize_network_dict
from Model.utils import *
from ComMethods import meth
SEED = 64
set_seed(SEED)


# =============================
# Data
# =============================
DATASET = 'Football'
# DATASET = 'Polbooks'
# DATASET = 'Cora'
# DATASET = 'Coauthor'
# DATASET = 'Youtube_C6'
# DATASET = 'Youtube_C8'

SAVE_ROOT = f"./Results/{DATASET}/benchmark/"
os.makedirs(SAVE_ROOT, exist_ok=True)

data = np.load(f'./Data/{DATASET}/network_with_label.npz')
A_noisy = data["Network"]
y_true = data["label"]

n_nodes = A_noisy.shape[0]
n_clusters = int(len(np.unique(y_true)))
print(f"[Data] DATASET={DATASET}, N={n_nodes}, n_clusters={n_clusters}")


RUNNERS = {
    "Noisy": lambda A: meth.noisy(A),
    "Katz": lambda A: meth.katz(A),
    "NE": lambda A: meth.NE(A),
    "CN": lambda A: meth.CN(A),
    "LP": lambda A: meth.LP(A),
    "NR": lambda A: meth.NR(A),
    "CLEAN": lambda A: meth.CLEAN(A, n_clusters=n_clusters),
}



# =============================
# Benchmark loop
# =============================
summary_rows = []
net_dict = {}

for method_name, runner in RUNNERS.items():
    print("=" * 80)
    print(f"[Run] {method_name}")

    method_dir = os.path.join(SAVE_ROOT, method_name)
    os.makedirs(method_dir, exist_ok=True)

    row = {
        "dataset": DATASET,
        "method": method_name,
    }

    # ---- run method ----
    W_pred = runner(A_noisy)
    W_pred = meth.ensure_symmetric_nonnegative(W_pred, zero_diag=True)
    net_dict[method_name] = W_pred

    # ---- matrix visualization ----
    visualize_matrices(
        matrices={f"{method_name} denoised graph": W_pred},
        y_true=y_true,
        save_path=os.path.join(method_dir, f"{method_name}_visualize_matrices.svg")
    )

    # ----- snr -----
    snr = snr_cohen_d(W_pred, y_true)
    row["community.snr"] = snr

    # ---- downstream clustering: compute and record ----
    clustering_ret = evaluate_single_network_clustering(
        A=W_pred,
        y_true=y_true,
        save_path=method_dir,
        random_state=SEED,
        mode='default'
    )

    row.update(flatten_clustering_results(clustering_ret, prefix="clustering"))

    # ---- simple matrix stats ----
    row["weight_sum"] = W_pred.sum()
    row["weight_mean"] = W_pred.mean()
    row["weight_max"] = W_pred.max()
    row["nnz"] = np.count_nonzero(W_pred)

    print(f"[Done] {method_name}")
    summary_rows.append(row)

print("=" * 80)
print("[All Done]")

visualize_network_dict(
    graphs_dict=net_dict,
    y_true=y_true,
    save_dir=os.path.join(SAVE_ROOT, "network")
)



# =============================
# 6) Save summary
# =============================
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("community.snr", ascending=False)
summary_csv = os.path.join(SAVE_ROOT, "benchmark_summary.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


from Model.plot_utils import plot_clustering_box_bench, plot_snr_bench
plot_snr_bench(summary_df, SAVE_ROOT)
plot_clustering_box_bench(summary_df, "ari", SAVE_ROOT)
plot_clustering_box_bench(summary_df, "nmi", SAVE_ROOT)

