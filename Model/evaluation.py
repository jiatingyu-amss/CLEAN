from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import os
from Model.clustering import run_single_clustering, ALGORITHMS


# ============================================================
# Downstream clustering ARI/NMI on noisy vs denoised
# ============================================================
def evaluate_downstream_clustering(
    A_noisy: np.ndarray,
    W_pred: np.ndarray,
    y_true: np.ndarray,
    algorithms: Optional[Sequence[str]] = None,
    random_state: int = 64,
    mode: str = "tune",
    save_path: str = None
):
    """
    Run downstream clustering on noisy and denoised graphs and report ARI/NMI.
    Returns a list of dict results for visualization.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n_clusters = len(np.unique(y_true))

    if algorithms is None:
        algorithms = ALGORITHMS

    print("==== Downstream clustering evaluation ====")

    results = []

    for algo in algorithms:
        # --- noisy ---
        z_noisy = run_single_clustering(
            A=A_noisy,
            algo=algo,
            n_clusters=n_clusters,
            random_state=random_state,
            mode=mode
        )

        # --- denoised ---
        z_den = run_single_clustering(
            A=W_pred,
            algo=algo,
            n_clusters=n_clusters,
            random_state=random_state,
            mode=mode
        )

        res = {
            "algo": algo,
            "ari_noisy": None,
            "nmi_noisy": None,
            "ari_den": None,
            "nmi_den": None,
        }

        # ---------- both skipped ----------
        if z_noisy is None and z_den is None:
            print(f"[Cluster-{algo}] skipped (both noisy and denoised are degenerate)")
            results.append(res)
            continue

        # ---------- noisy skipped ----------
        if z_noisy is None:
            ari_den = adjusted_rand_score(y_true, z_den)
            nmi_den = normalized_mutual_info_score(y_true, z_den)
            print(
                f"[Cluster-{algo:16s}] "
                f"Noisy   : SKIPPED | "
                f"Denoised: ARI={ari_den:.4f}, NMI={nmi_den:.4f}"
            )
            res["ari_den"] = ari_den
            res["nmi_den"] = nmi_den
            results.append(res)
            continue

        # ---------- denoised skipped ----------
        if z_den is None:
            ari_noisy = adjusted_rand_score(y_true, z_noisy)
            nmi_noisy = normalized_mutual_info_score(y_true, z_noisy)
            print(
                f"[Cluster-{algo:16s}] "
                f"Noisy   : ARI={ari_noisy:.4f}, NMI={nmi_noisy:.4f} | "
                f"Denoised: SKIPPED"
            )
            res["ari_noisy"] = ari_noisy
            res["nmi_noisy"] = nmi_noisy
            results.append(res)
            continue

        # ---------- normal case ----------
        ari_noisy = adjusted_rand_score(y_true, z_noisy)
        nmi_noisy = normalized_mutual_info_score(y_true, z_noisy)
        ari_den = adjusted_rand_score(y_true, z_den)
        nmi_den = normalized_mutual_info_score(y_true, z_den)

        print(
            f"[Cluster-{algo:16s}] "
            f"Noisy   : ARI={ari_noisy:.4f}, NMI={nmi_noisy:.4f} | "
            f"Denoised: ARI={ari_den:.4f}, NMI={nmi_den:.4f}"
        )

        res["ari_noisy"] = ari_noisy
        res["nmi_noisy"] = nmi_noisy
        res["ari_den"] = ari_den
        res["nmi_den"] = nmi_den
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df.index = results_df['algo']
    results_df = results_df.drop(columns=['algo'])
    results_df.to_csv(save_path + 'results_df.csv')

    return results

def evaluate_single_network_clustering(
    A: np.ndarray,
    y_true: np.ndarray,
    algorithms: Optional[Sequence[str]] = None,
    random_state: int = 64,
    mode: str = 'tune',
    save_path: str = None,
    file_name: str = "clustering_results.csv",
):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n_clusters = len(np.unique(y_true))

    if algorithms is None:
        algorithms = ALGORITHMS

    print("==== Single-network downstream clustering evaluation ====")

    results = []

    for algo in algorithms:
        z = run_single_clustering(
            A=A,
            algo=algo,
            n_clusters=n_clusters,
            random_state=random_state,
            mode=mode
        )

        res = {
            "algo": algo,
            "ari": None,
            "nmi": None,
        }

        if z is None:
            print(f"[Cluster-{algo:16s}] SKIPPED")
            results.append(res)
            continue

        ari = adjusted_rand_score(y_true, z)
        nmi = normalized_mutual_info_score(y_true, z)

        print(f"[Cluster-{algo:16s}] ARI={ari:.4f}, NMI={nmi:.4f}")

        res["ari"] = ari
        res["nmi"] = nmi
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df.index = results_df["algo"]
    results_df = results_df.drop(columns=["algo"])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        results_df.to_csv(os.path.join(save_path, file_name))

    return results



# ============================================================
# Metrics (for SBM)
# ============================================================
def _upper_tri_labels_and_scores(M, y_true):
    y = y_true.astype(np.int32)
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    i, j = iu
    y_bin = (y[i] == y[j]).astype(np.int32)
    s = M[iu].astype(np.float32)
    return y_bin, s

def snr_(M, y_true, eps=1e-6, in_db=True):
    y_bin, s = _upper_tri_labels_and_scores(M, y_true)
    intra = s[y_bin == 1]
    inter = s[y_bin == 0]
    mu_intra = float(np.mean(intra)) if intra.size else 0.0
    mu_inter = float(np.mean(inter)) if inter.size else 0.0
    ratio = (mu_intra + eps) / (mu_inter + eps)
    if in_db:
        return float(10.0 * np.log10(ratio))
    return float(ratio)

def snr_cohen_d(M, y_true, eps=1e-6):
    y_bin, s = _upper_tri_labels_and_scores(M, y_true)

    intra = s[y_bin == 1]
    inter = s[y_bin == 0]
    mu_intra = np.mean(intra)
    mu_inter = np.mean(inter)
    std_intra = np.std(intra)
    std_inter = np.std(inter)

    pooled_std = np.sqrt((std_intra**2 + std_inter**2) / 2) + eps
    d = (mu_intra - mu_inter) / pooled_std

    return float(d)

def modularity_weighted(W, y):
    k = W.sum(axis=1)
    m2 = float(k.sum())
    if m2 <= 0:
        return np.nan

    Q = 0.0
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if idx.size <= 1:
            continue
        W_cc = W[np.ix_(idx, idx)].sum()
        k_c = k[idx].sum()
        Q += (W_cc - (k_c * k_c) / m2)

    Q = Q / m2
    return float(Q)

def evaluate_sbm_metrics(W_pred, y_true, data_name, snr_db=True):
    W_pred = (W_pred + W_pred.T)/2
    np.fill_diagonal(W_pred, 0.0)
    out = {}
    out[f"{data_name}_modularity"] = modularity_weighted(W_pred, y_true)
    out[f"{data_name}_snr"] = snr_(W_pred, y_true, in_db=snr_db)
    out[f"{data_name}_nsnr"] = snr_cohen_d(W_pred, y_true)
    return out
