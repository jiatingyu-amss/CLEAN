from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import scipy.sparse as sp



# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int = 0) -> None:
    import os, random
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



# -----------------------------
# Graph utilities
# -----------------------------
def adj_to_edge_list(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a dense (0/1) adjacency to undirected edge list (upper-tri only)."""
    iu = np.triu_indices(A.shape[0], k=1)
    m = A[iu] > 0
    i = iu[0][m].astype(np.int64)
    j = iu[1][m].astype(np.int64)
    return i, j


def edge_list_to_csr(n: int, i_idx: np.ndarray, j_idx: np.ndarray, undirected: bool = True) -> sp.csr_matrix:
    """Build CSR adjacency from edge list."""
    i = i_idx.astype(np.int64)
    j = j_idx.astype(np.int64)
    data = np.ones_like(i, dtype=np.float32)

    A = sp.coo_matrix((data, (i, j)), shape=(n, n)).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()

    if undirected:
        A = A.maximum(A.T)
        A.setdiag(0.0)
        A.eliminate_zeros()
    return A


def build_consensus_network(cand) -> np.ndarray:
    """
    Build a symmetric network matrix from coordinate-style candidate edges.
    """
    import numpy as np
    import scipy.sparse as sp
    i_idx = cand.i
    j_idx = cand.j
    scores = cand.s_cons
    n_nodes = int(max(i_idx.max(), j_idx.max()) + 1)

    rows = np.concatenate([i_idx, j_idx])
    cols = np.concatenate([j_idx, i_idx])
    weights = np.concatenate([scores, scores])

    mat = sp.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for r, c, v in zip(rows, cols, weights):
        if W[r, c] < v:
            W[r, c] = v
            W[c, r] = v  # Force symmetry

    np.fill_diagonal(W, 0.0)
    return W



# -----------------------------
# Candidate edges and features
# -----------------------------
@dataclass
@dataclass
class EdgeScoreBank:
    deg: np.ndarray | None = None
    a: np.ndarray | None = None
    cn: np.ndarray | None = None
    aa: np.ndarray | None = None
    a3: np.ndarray | None = None
    ra: np.ndarray | None = None
    lp: np.ndarray | None = None
    cos: np.ndarray | None = None
    pa: np.ndarray | None = None
    jaccard: np.ndarray | None = None
    salton: np.ndarray | None = None
    sp_inv: np.ndarray | None = None

    def get(self, name: str, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        feat = getattr(self, name)
        if feat is None:
            raise KeyError(f"Score '{name}' not found in EdgeScoreBank.")

        if name == "deg":
            di = feat[i].astype(np.float32)
            dj = feat[j].astype(np.float32)
            return np.stack([di, dj], axis=1)

        return feat[i, j].astype(np.float32)

def build_edge_score_bank(
    A: sp.csr_matrix,
    emb: np.ndarray | None,
    methods: list[str] = [
        "deg", "a", "cn", "a3", "aa", "ra", "lp", "cos",
        "pa", "jaccard", "salton", "sp_inv"
    ],
    lp_beta: float = 1 / 1.5,
    eps: float = 1e-12,
) -> "EdgeScoreBank":
    """
    Build a bank of pairwise structural scores.

    Supported methods:
      - "deg": node degree vector; edge feature becomes [d_i, d_j]
      - "a": adjacency
      - "cn": common neighbors = A^2
      - "a3": A^3
      - "aa": Adamic-Adar
      - "ra": Resource Allocation
      - "lp": Local Path
      - "cos": embedding cosine
      - "pa": Preferential Attachment = deg_i * deg_j
      - "jaccard": |N(i)∩N(j)| / |N(i)∪N(j)|
      - "salton": |N(i)∩N(j)| / sqrt(deg_i * deg_j)
      - "sp_inv": 1 / (shortest_path_distance + 1)
    """
    from scipy.sparse.csgraph import shortest_path

    methods_set = {m.lower() for m in methods}

    # ---- dense adjacency ----
    A_dense = A.toarray().astype(np.float64, copy=False)

    # undirected safety
    np.fill_diagonal(A_dense, 0.0)
    A_dense = np.maximum(A_dense, A_dense.T)

    bank: dict[str, np.ndarray] = {}

    # ---- degrees ----
    deg = A_dense.sum(axis=1).astype(np.float64)

    if "deg" in methods_set:
        bank["deg"] = np.nan_to_num(deg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- precompute powers ----
    need_A2 = any(m in methods_set for m in ["cn", "lp", "aa", "ra", "jaccard", "salton"])
    need_A3 = ("lp" in methods_set) or ("a3" in methods_set)

    A2 = None
    A3 = None
    if need_A2:
        A2 = A_dense @ A_dense
    if need_A3:
        A3 = A2 @ A_dense

    # ---- adjacency ----
    if "a" in methods_set:
        bank["a"] = A_dense.astype(np.float32)

    # ---- CN ----
    if "cn" in methods_set:
        bank["cn"] = A2.astype(np.float32)

    # ---- AA ----
    if "aa" in methods_set:
        deg_aa = np.maximum(deg, 2.0)
        inv_log = 1.0 / np.log(deg_aa + eps)
        Aw = A_dense * inv_log[np.newaxis, :]
        bank["aa"] = (Aw @ A_dense.T).astype(np.float32)

    # ---- RA ----
    if "ra" in methods_set:
        deg_ra = np.maximum(deg, 1.0)
        inv_deg = 1.0 / (deg_ra + eps)
        Aw = A_dense * inv_deg[np.newaxis, :]
        bank["ra"] = (Aw @ A_dense.T).astype(np.float32)

    # ---- A^3 ----
    if "a3" in methods_set:
        bank["a3"] = A3.astype(np.float32)

    # ---- LP ----
    if "lp" in methods_set:
        bank["lp"] = (
            lp_beta * A_dense +
            (lp_beta ** 2) * A2 +
            (lp_beta ** 3) * A3
        ).astype(np.float32)

    # ---- embedding cosine ----
    if "cos" in methods_set and emb is not None:
        x = emb.astype(np.float32)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        bank["cos"] = (x @ x.T).astype(np.float32)

    # ---- Preferential Attachment ----
    if "pa" in methods_set:
        bank["pa"] = np.outer(deg, deg).astype(np.float32)

    # ---- Jaccard ----
    if "jaccard" in methods_set:
        cn = A2
        deg_col = deg[:, None]
        deg_row = deg[None, :]
        union = deg_col + deg_row - cn
        union = np.maximum(union, eps)
        jac = cn / union
        bank["jaccard"] = jac.astype(np.float32)

    # ---- Salton / cosine-over-neighborhoods ----
    if "salton" in methods_set:
        cn = A2
        denom = np.sqrt(np.outer(deg, deg))
        denom = np.maximum(denom, eps)
        salton = cn / denom
        bank["salton"] = salton.astype(np.float32)

    # ---- Inverse shortest-path distance ----
    if "sp_inv" in methods_set:
        dist = shortest_path(
            csgraph=A,
            directed=False,
            unweighted=True,
            return_predecessors=False,
        ).astype(np.float64)

        sp_inv = np.zeros_like(dist, dtype=np.float64)
        finite_mask = np.isfinite(dist)
        sp_inv[finite_mask] = 1.0 / (dist[finite_mask] + 1.0)
        np.fill_diagonal(sp_inv, 0.0)
        bank["sp_inv"] = sp_inv.astype(np.float32)

    # ---- sanitize ----
    for k, v in list(bank.items()):
        if k == "deg":
            continue
        v = np.asarray(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = 0.5 * (v + v.T)
        np.fill_diagonal(v, 0.0)
        bank[k] = v

    return EdgeScoreBank(
        deg=bank.get("deg", None),
        a=bank.get("a", None),
        cn=bank.get("cn", None),
        aa=bank.get("aa", None),
        a3=bank.get("a3", None),
        ra=bank.get("ra", None),
        lp=bank.get("lp", None),
        cos=bank.get("cos", None),
        pa=bank.get("pa", None),
        jaccard=bank.get("jaccard", None),
        salton=bank.get("salton", None),
        sp_inv=bank.get("sp_inv", None),
    )


def expand_edges_by_embedding(
    A: sp.csr_matrix,
    bank: EdgeScoreBank,
    expand_ratio: float,
    min_expand: int = 1,
    max_expand: int | None = None,
):
    sim = bank.cos
    assert sim is not None, "cos similarity required"

    n = A.shape[0]
    A_dense = A.toarray()
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(int)

    top_i, top_j = [], []

    for i in range(n):
        d = deg[i]
        k = max(min_expand, int(np.ceil(expand_ratio * max(d, 1))))
        if max_expand is not None:
            k = min(k, max_expand)

        mask = (A_dense[i] == 0)
        scores = sim[i].copy()
        scores[~mask] = -np.inf

        if np.all(scores == -np.inf):
            continue

        idx = np.argsort(-scores)[:k]
        for j in idx:
            top_i.append(i)
            top_j.append(j)

    return np.array(top_i), np.array(top_j)


def expand_edges_by_local(
    A: sp.csr_matrix,
    bank: EdgeScoreBank,
    method: str,
    expand_ratio: float,
):
    mat = getattr(bank, method)
    assert mat is not None, f"{method} not in bank"

    n = A.shape[0]
    A_dense = A.toarray()
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(int)

    top_i, top_j = [], []

    for i in range(n):
        d = deg[i]
        k = max(1, int(np.ceil(expand_ratio * max(d, 1))))

        mask = (A_dense[i] == 0)
        scores = mat[i].copy()
        scores[~mask] = -np.inf

        if np.all(scores == -np.inf):
            continue

        idx = np.argsort(-scores)[:k]
        for j in idx:
            top_i.append(i)
            top_j.append(j)

    return np.array(top_i), np.array(top_j)



def expand_edges(
    A: sp.csr_matrix,
    bank: EdgeScoreBank,
    expand_ratio: float = 3,
):
    i1, j1 = expand_edges_by_embedding(A, bank, expand_ratio=expand_ratio)
    i2, j2 = expand_edges_by_local(A, bank, method="cn", expand_ratio=expand_ratio)

    i = np.concatenate([i1, i2])
    j = np.concatenate([j1, j2])

    return i, j



@dataclass
class CandidateEdges:
    i: np.ndarray
    j: np.ndarray
    s_cons: np.ndarray


def build_candidate_edges(
    E_i: np.ndarray,
    E_j: np.ndarray,
    top_i: np.ndarray,
    top_j: np.ndarray,
    partitions: list[np.ndarray],
    consensus_thr: float = 0.2,
) -> CandidateEdges:
    """
    Build initial candidate pool as the union of original edges and structural edges,
    then keep only edges with consensus score > consensus_thr.
    """

    def _canon(p: int, q: int) -> tuple[int, int]:
        return (p, q) if p < q else (q, p)

    # Step 1: initial candidate pool = original edges + structural edges
    edge_set: set[tuple[int, int]] = set()

    for a, b in zip(E_i, E_j):
        if a != b:
            edge_set.add(_canon(int(a), int(b)))

    for a, b in zip(top_i, top_j):
        if a != b:
            edge_set.add(_canon(int(a), int(b)))

    if not edge_set:
        return CandidateEdges(
            i=np.zeros(0, dtype=np.int64),
            j=np.zeros(0, dtype=np.int64),
            s_cons=np.zeros(0, dtype=np.float32),
        )

    edges = np.array(sorted(edge_set), dtype=np.int64)
    i_all = edges[:, 0]
    j_all = edges[:, 1]

    # Step 2: compute consensus on the whole initial pool
    if len(partitions) == 0:
        s_cons_all = np.zeros(len(i_all), dtype=np.float32)
    else:
        M = len(partitions)
        same = np.zeros(len(i_all), dtype=np.float32)
        for z in partitions:
            z = np.asarray(z)
            same += (z[i_all] == z[j_all]).astype(np.float32)
        s_cons_all = (same / float(M)).astype(np.float32)

    # Step 3: keep only edges with consensus > consensus_thr
    keep = s_cons_all > float(consensus_thr)

    return CandidateEdges(
        i=i_all[keep].astype(np.int64),
        j=j_all[keep].astype(np.int64),
        s_cons=s_cons_all[keep].astype(np.float32),
    )


@dataclass
class EdgeFeatures:
    X: np.ndarray
    y: np.ndarray


def build_edge_features(
    cand: CandidateEdges,
    bank: EdgeScoreBank,
    feature_list: list[str] = [
        "deg", "a", "cn", "aa", "ra", "a3", "cos",
        "pa", "jaccard", "salton", "sp_inv"
    ],
    add_random: bool = False,
    random_std: float = 1.0,
    random_seed: int | None = None,
    pos_thr: float = 0.5,
):
    """
    Build edge features for all upper-triangle node pairs.

    Labels:
      1  : candidate edge with s_cons >= pos_thr
      -1 : candidate edge with s_cons < pos_thr
      0  : non-candidate edge

    Also add consensus score s_ij as an input feature.
    """

    # ---- infer number of nodes ----
    n_nodes = None
    for name in ["a", "cn", "aa", "ra", "a3", "lp", "cos"]:
        mat = getattr(bank, name, None)
        if mat is not None:
            n_nodes = int(mat.shape[0])
            break

    if n_nodes is None and getattr(bank, "deg", None) is not None:
        n_nodes = int(len(bank.deg))

    if n_nodes is None:
        raise ValueError("Cannot infer n_nodes from EdgeScoreBank.")

    # ---- all upper-triangle pairs ----
    all_i, all_j = np.triu_indices(n_nodes, k=1)
    all_i = all_i.astype(np.int64)
    all_j = all_j.astype(np.int64)
    E_all = len(all_i)

    # ---- label initialization ----
    y = np.zeros(E_all, dtype=np.float32)

    # ---- consensus feature initialization ----
    sij_feat = np.zeros(E_all, dtype=np.float32)

    # map edge -> index
    pair_to_idx = {(int(i), int(j)): idx for idx, (i, j) in enumerate(zip(all_i, all_j))}

    # fill candidate edges
    for i, j, s in zip(cand.i, cand.j, cand.s_cons):
        key = (int(i), int(j)) if i < j else (int(j), int(i))
        idx = pair_to_idx[key]

        sij_feat[idx] = s

        if s >= pos_thr:
            y[idx] = 1
        else:
            y[idx] = -1

    # ---- build feature matrix ----
    feats = []

    # add sij as feature (VERY IMPORTANT)
    feats.append(sij_feat[:, None])

    for name in feature_list:
        f = bank.get(name, all_i, all_j)
        f = np.asarray(f, dtype=np.float32)

        if f.ndim == 1:
            f = f[:, None]

        feats.append(f)

    # optional random feature
    if add_random:
        rng = np.random.default_rng(random_seed)
        rand_feat = rng.normal(
            loc=0.0,
            scale=random_std,
            size=(E_all, 1),
        ).astype(np.float32)

        feats.append(rand_feat)

    X = np.concatenate(feats, axis=1).astype(np.float32)

    # feature normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return EdgeFeatures(X=X, y=y)


# -----------------------------
# Node embedding
# -----------------------------
def node2vec_embedding_from_adjacency(
    A: np.ndarray,
    dim: int = 32,
    walk_length: int = 40,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 128,
    seed: int = 64,
) -> np.ndarray:
    import os
    import networkx as nx
    from node2vec import Node2Vec

    # -----------------------------
    # Determinism / seeds
    # -----------------------------
    # 1) python hash seed (affects dict/set iteration order in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) limit thread pools that may introduce nondeterministic summations
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    workers = 1

    print("[Embedding] Running node2vec......")

    A_np = A.astype(np.float32)
    n = A_np.shape[0]

    # Build an undirected weighted graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Use upper triangle as edge list (stable order)
    iu = np.triu_indices(n, k=1)
    w_all = A_np[iu]
    mask = w_all > 0
    ii = iu[0][mask]
    jj = iu[1][mask]
    ww = w_all[mask].astype(np.float64)

    # (optional) sort edges to be extra safe about insertion order
    # lexsort by (u, v)
    if ii.size > 0:
        order = np.lexsort((jj, ii))
        ii = ii[order]
        jj = jj[order]
        ww = ww[order]

    for u, v, w in zip(ii.tolist(), jj.tolist(), ww.tolist()):
        G.add_edge(int(u), int(v), weight=float(w))

    node2vec = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        weight_key="weight",
        quiet=True,
    )

    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=batch_words,
        seed=seed,
        workers=workers,  
    )

    emb = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        key = str(i)
        emb[i] = model.wv[key]

    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


# ------------------ablation ---------------------
def shuffle_partitions(partitions: list[np.ndarray], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = []
    for z in partitions:
        z = np.asarray(z)
        shuffled.append(rng.permutation(z))
    return shuffled



def flatten_clustering_results(results, prefix="clustering"):
    out = {}
    if results is None:
        return out

    for item in results:
        algo = item.get("algo", "unknown")
        ari = item.get("ari", None)
        nmi = item.get("nmi", None)
        out[f"{prefix}.{algo}.ari"] = np.nan if ari is None else float(ari)
        out[f"{prefix}.{algo}.nmi"] = np.nan if nmi is None else float(nmi)
    return out



def remove_smallest_k_comm(A, y, k):
    import numpy as np
    from collections import Counter
    from scipy.sparse import issparse, csr_matrix
    from scipy.sparse.csgraph import connected_components

    """
    Remove nodes belonging to the k smallest communities, then keep only
    the largest connected component (LCC) of the remaining graph.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix
        Adjacency matrix of shape (N, N).
    y : array-like
        Community labels of shape (N,).
    k : int
        Number of smallest communities to remove.

    Returns
    -------
    A_new : np.ndarray or scipy.sparse matrix
        Adjacency matrix of the final subgraph.
    y_new : np.ndarray
        Labels of nodes in the final subgraph.
    """
    y = np.asarray(y)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if A.shape[0] != len(y):
        raise ValueError("A and y must have the same number of nodes.")
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")

    # Count the size of each community
    label_counts = Counter(y)

    if k > len(label_counts):
        raise ValueError(f"k={k} is larger than the number of communities ({len(label_counts)}).")

    # Sort communities by size ascending, then by label for deterministic behavior
    sorted_labels = sorted(label_counts.items(), key=lambda x: (x[1], x[0]))
    smallest_labels = [label for label, _ in sorted_labels[:k]]

    # Remove nodes in the smallest k communities
    keep_mask_stage1 = ~np.isin(y, smallest_labels)
    keep_idx_stage1 = np.where(keep_mask_stage1)[0]

    if len(keep_idx_stage1) == 0:
        raise ValueError("No nodes remain after removing the smallest k communities.")

    # Build the intermediate subgraph
    if issparse is not None and issparse(A):
        A_stage1 = A[keep_idx_stage1][:, keep_idx_stage1]
    else:
        A_stage1 = A[np.ix_(keep_idx_stage1, keep_idx_stage1)]

    y_stage1 = y[keep_idx_stage1]

    # Find the largest connected component
    if issparse is not None and issparse(A_stage1):
        n_components, component_labels = connected_components(
            csgraph=A_stage1, directed=False, return_labels=True
        )
    else:
        # Convert dense matrix to sparse for connected component computation
        if connected_components is None or csr_matrix is None:
            raise ImportError("scipy is required for connected component computation.")
        A_stage1_sparse = csr_matrix(A_stage1)
        n_components, component_labels = connected_components(
            csgraph=A_stage1_sparse, directed=False, return_labels=True
        )

    # Identify the largest connected component
    comp_sizes = np.bincount(component_labels)
    largest_comp = np.argmax(comp_sizes)

    keep_mask_stage2 = (component_labels == largest_comp)
    keep_idx_stage2 = np.where(keep_mask_stage2)[0]

    # Final node indices in the original graph
    final_idx = keep_idx_stage1[keep_idx_stage2]

    # Build the final subgraph
    if issparse is not None and issparse(A_stage1):
        A_new = A_stage1[keep_idx_stage2][:, keep_idx_stage2]
    else:
        A_new = A_stage1[np.ix_(keep_idx_stage2, keep_idx_stage2)]

    y_new = y_stage1[keep_idx_stage2]

    return A_new, y_new



def connect_graph_with_tiny_edges(A, tiny_weight=1e-6, random_state=42):
    import numpy as np
    from scipy.sparse import issparse, csr_matrix
    from scipy.sparse.csgraph import connected_components
    """
    Connect an undirected graph by adding tiny-weight edges between connected components.

    If the graph is already connected, the function returns a copy of A unchanged.
    If the graph has multiple connected components, it adds the minimum number of
    undirected edges needed to make the graph connected.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix
        Symmetric adjacency matrix of shape (N, N).
    tiny_weight : float, default=1e-6
        Weight assigned to newly added edges.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    A_new : np.ndarray or scipy.sparse matrix
        Connected adjacency matrix.
    added_edges : list of tuple, optional
        List of added edges in the form (i, j, tiny_weight), where i < j.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    if tiny_weight <= 0:
        raise ValueError("tiny_weight must be positive.")

    rng = np.random.default_rng(random_state)

    # Convert to sparse format for connected component computation
    if issparse is not None and issparse(A):
        A_cc = A.tocsr()
    else:
        if csr_matrix is None or connected_components is None:
            raise ImportError("scipy is required for connected component computation.")
        A_cc = csr_matrix(A)

    # Find connected components
    n_components, labels = connected_components(
        csgraph=A_cc,
        directed=False,
        return_labels=True
    )

    # Return early if the graph is already connected
    if n_components == 1:
        return A.copy()

    # Collect node indices for each component
    components = [np.where(labels == comp_id)[0] for comp_id in range(n_components)]

    # Shuffle component order to randomize the connection pattern
    order = rng.permutation(n_components)
    components = [components[idx] for idx in order]

    added_edges = []

    if issparse is not None and issparse(A):
        # Use LIL format for efficient incremental assignment
        A_new = A.tolil(copy=True)

        # Connect components in a chain with n_components - 1 edges
        for t in range(n_components - 1):
            comp_u = components[t]
            comp_v = components[t + 1]

            i = rng.choice(comp_u)
            j = rng.choice(comp_v)

            # Add symmetric edge for an undirected graph
            A_new[i, j] = tiny_weight
            A_new[j, i] = tiny_weight

            added_edges.append((int(min(i, j)), int(max(i, j)), float(tiny_weight)))

        A_new = A_new.tocsr()

    else:
        A_new = np.array(A, copy=True)

        # Connect components in a chain with n_components - 1 edges
        for t in range(n_components - 1):
            comp_u = components[t]
            comp_v = components[t + 1]

            i = rng.choice(comp_u)
            j = rng.choice(comp_v)

            # Add symmetric edge for an undirected graph
            A_new[i, j] = tiny_weight
            A_new[j, i] = tiny_weight

            added_edges.append((int(min(i, j)), int(max(i, j)), float(tiny_weight)))

    return A_new
