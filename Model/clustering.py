from __future__ import annotations
from typing import Optional, Sequence
import os
import numpy as np


ALGORITHMS = [
    "igraph_louvain",
    "igraph_leiden",
    "igraph_infomap",
    "igraph_label_prop",
    "igraph_walktrap",
    "igraph_fast_greedy",
    "igraph_spinglass",
    "sklearn_spectral",
]


def run_igraph_via_rscript(
    A: np.ndarray,
    method: str,
    seed: int,
    n_clusters: int,
    mode: str = "tune",
    rscript_path: str = "Rscript",
    r_bridge_path: str = './Model/igraph_bridge.R',
) -> np.ndarray | None:
    import subprocess
    import tempfile
    """
    Run R igraph clustering via an external Rscript.
    Returns 0-based labels or None on failure.
    """
    method = method.lower().strip()
    n = A.shape[0]

    # Build edge list (upper triangle)
    iu = np.triu_indices(n, k=1)
    w = A[iu]
    mask = w > 0
    i_idx = iu[0][mask].astype(np.int64)
    j_idx = iu[1][mask].astype(np.int64)
    w_idx = w[mask].astype(np.float32)

    try:
        with tempfile.TemporaryDirectory() as td:
            edge_csv = os.path.join(td, "edges.csv")
            out_csv = os.path.join(td, "labels.csv")

            # Write edges.csv with columns i,j,w (0-based)
            with open(edge_csv, "w", encoding="utf-8") as f:
                f.write("i,j,w\n")
                for ii, jj, ww in zip(i_idx, j_idx, w_idx):
                    f.write(f"{int(ii)},{int(jj)},{float(ww)}\n")

            cmd = [
                rscript_path,
                r_bridge_path,
                mode,
                edge_csv,
                str(int(n)),
                method,
                str(int(seed)),
                str(int(n_clusters)),
                out_csv,
            ]

            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if p.returncode != 0:
                err = p.stderr.decode(errors="replace")
                print(f"[Rscript-igraph] failed ({method}), returncode={p.returncode}\n{err}")
                return None

            # Read labels.csv (single column: label)
            labels = []
            with open(out_csv, "r", encoding="utf-8", errors="replace") as f:
                _ = f.readline()  # header
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # CSV: label
                    labels.append(int(line.split(",")[0]))
            labels = np.asarray(labels, dtype=np.int64)

            if labels.size != n:
                print(f"[Rscript-igraph] invalid labels length: got {labels.size}, expected {n}")
                return None

            return labels

    except Exception as e:
        print(f"[Rscript-igraph] exception ({method}): {repr(e)}")
        return None



def run_single_clustering(
    A: np.ndarray,
    algo: str,
    n_clusters: int,
    random_state: int = 0,
    mode: str = 'tune',
    # Quality gate
    min_clusters: int = 2,
    max_single_ratio: float = 0.9,
) -> Optional[np.ndarray]:
    """
    Run one clustering algorithm and return labels (N,).
    Returns: labels or None if degenerate / unavailable.
    Notes:
      - Input A is treated as an undirected weighted adjacency.
      - A is symmetrized and diagonal is zeroed for safety.
      - Quality gate filters: too few clusters or one giant cluster.
    """

    # print(f'[Ensemble] ========f{algo}============')
    algo = algo.lower().strip()

    # -------------------------
    # Sanitize adjacency
    # -------------------------
    A = np.asarray(A, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, A.T)
    N = A.shape[0]

    # -------------------------
    # Helper: quality control
    # -------------------------
    def _quality_gate(z: np.ndarray) -> Optional[np.ndarray]:
        z = np.asarray(z, dtype=np.int64)
        uniq, cnt = np.unique(z, return_counts=True)
        if len(uniq) < min_clusters:
            return None
        if (cnt.max() / cnt.sum()) > max_single_ratio:
            return None
        return z.astype(int)

    # ============================================================
    # R igraph
    # ============================================================
    igraph_alias = {
        "igraph_louvain": "louvain",
        "igraph_leiden": "leiden",
        "igraph_infomap": "infomap",
        "igraph_label_prop": "label_prop",
        "igraph_walktrap": "walktrap",
        "igraph_fast_greedy": "fast_greedy",
        "igraph_leading_eigen": "leading_eigen",
        "igraph_spinglass": "spinglass",
    }

    algo_igraph = igraph_alias.get(algo, None)
    if algo_igraph is not None:
        z = run_igraph_via_rscript(
            A=A,
            method=algo_igraph,
            seed=random_state,
            n_clusters=n_clusters,
            mode = mode,
        )
        if z is None:
            return None
        return _quality_gate(z)

    # ============================================================
    # Python: spectral
    # ============================================================
    if algo in ["spectral", "sklearn_spectral"]:
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        z = sc.fit_predict(A)
        return _quality_gate(z)

    else:
        return None



def build_partition_ensemble(
    A_noisy: np.ndarray,
    n_clusters: int,
    algorithms: Optional[Sequence[str]] = None,
    random_state: int = 64,
    mode: str = "tune",
):
    """
    Produce multiple partitions for pseudo labels
    """
    if algorithms is None:
        algorithms = ALGORITHMS

    parts = []
    for algo in algorithms:
        z = run_single_clustering(
            A=A_noisy,
            algo=algo,
            n_clusters=n_clusters,
            random_state=random_state,
            mode=mode
        )
        if z is not None:
            parts.append(z)
    print(f"[Ensemble] kept {len(parts)}/{len(algorithms)} partitions (after quality gate)")

    return parts, algorithms


