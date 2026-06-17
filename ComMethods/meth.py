import numpy as np
from numpy.linalg import eigvals, solve, LinAlgError


def ensure_symmetric_nonnegative(W, zero_diag=True):
    W = np.asarray(W, dtype=np.float64)
    W = 0.5 * (W + W.T)
    W[W < 0] = 0.0
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    W = W/W.max()
    return W

def NR(mat, m=1.5):
    N = mat.shape[0]
    P1 = mat / mat.sum(1).reshape(-1, 1)
    P2 = (m - 1) * P1.dot(np.linalg.inv(m * np.eye(N) - P1))
    w_out = np.diag(mat.sum(1)).dot(P2)
    return w_out

def katz(A):
    from numpy.linalg import eigvals, solve
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    I = np.eye(n, dtype=np.float64)

    eigs = eigvals(A)
    r = float(np.max(np.abs(eigs)))

    A_use = A / r
    r_use = 1.0

    beta = 1.0 / 2.0
    threshold = 1.0 / r_use
    beta_eff = beta if beta < threshold else threshold / 2.0

    w_out = solve(I - beta_eff * A_use, I) - I
    return np.asarray(w_out, dtype=np.float64)

def CN(A):
    w_out = A @ A
    return w_out


def LP(A):
    A2 = A @ A
    A3 = A2 @ A
    w_out = A3 + A2
    return w_out



# ======================NE ======================NE
def _ne_dn(w: np.ndarray, mode: str = "ave", eps: float = 1e-12) -> np.ndarray:
    """
    Python rewrite of core/NE_dn.m

    MATLAB:
        w = w * length(w);
        D = sum(abs(w),2) + eps;
        if type == 'ave'
            D = 1./D; wn = D*w
        elseif type == 'gph'
            D = 1./sqrt(D); wn = D*(w*D)
    """
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    w = w * n

    d = np.sum(np.abs(w), axis=1) + eps

    if mode == "ave":
        return (1.0 / d)[:, None] * w
    elif mode == "gph":
        s = 1.0 / np.sqrt(d)
        return s[:, None] * w * s[None, :]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _dominateset(aff_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Python rewrite of core/dominateset.m

    Keep top-k entries in each row, then symmetrize by averaging:
        PNN_matrix = (PNN_matrix1 + PNN_matrix1') / 2
    """
    A = np.asarray(aff_matrix, dtype=np.float64)
    n = A.shape[0]
    if k <= 0:
        return np.zeros_like(A)
    k = min(k, n)

    out = np.zeros_like(A)

    # top-k indices per row (descending)
    # argpartition is faster than full argsort
    idx_part = np.argpartition(-A, kth=k-1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    vals = A[row_idx, idx_part]
    out[row_idx, idx_part] = vals

    return 0.5 * (out + out.T)


def _transition_fields(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Python rewrite of core/TransitionFields.m

    MATLAB:
        zeroindex = find(sum(W,2)==0);
        W = W*length(W);
        W = NE_dn(W,'ave');
        w = sqrt(sum(abs(W))+eps);
        W = W./repmat(w,length(W),1);
        W = W*W';
        Wnew = W;
        Wnew(zeroindex,:) = 0;
        Wnew(:,zeroindex) = 0;
    """
    W = np.asarray(W, dtype=np.float64)
    zeroindex = np.where(np.sum(W, axis=1) == 0)[0]

    W = W * W.shape[0]
    W = _ne_dn(W, mode="ave", eps=eps)

    # MATLAB sum(abs(W)) is column sum
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / w[None, :]
    Wnew = W @ W.T

    if zeroindex.size > 0:
        Wnew[zeroindex, :] = 0.0
        Wnew[:, zeroindex] = 0.0

    return Wnew


def NE(
    W_in: np.ndarray,
    order: float = 2.0,
    K: int | None = None,
    alpha: float = 0.9,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Python rewrite of wangboyunze/Network_Enhancement/core/Network_Enhancement.m

    Parameters
    ----------
    W_in : (N, N) array
        Input symmetric weighted network.
    order : float
        Diffusion order. Typical values in repo comment: 0.5, 1, 2.
    K : int or None
        Number of neighbors. Default matches MATLAB:
        min(20, ceil(N/10))
    alpha : float
        Regularization parameter.
    eps : float
        Numerical stability constant.

    Returns
    -------
    W_out : (N, N) array
        Enhanced network.
    """
    W_in = np.asarray(W_in, dtype=np.float64)

    if W_in.ndim != 2 or W_in.shape[0] != W_in.shape[1]:
        raise ValueError("W_in must be a square matrix.")

    n = W_in.shape[0]
    if K is None:
        K = min(20, int(np.ceil(n / 10.0)))

    # remove diagonal
    W_in1 = W_in * (1.0 - np.eye(n))

    # MATLAB: zeroindex = find(sum(abs(W_in1))>0)
    # sum(abs(W_in1)) is column sum, but for symmetric matrices row/col are same.
    keep = np.where(np.sum(np.abs(W_in1), axis=0) > 0)[0]

    W_out = np.zeros_like(W_in)
    if keep.size == 0:
        return W_out

    W0 = W_in[np.ix_(keep, keep)]

    # W = NE_dn(W0, 'ave'); W = (W + W') / 2;
    W = _ne_dn(W0, mode="ave", eps=eps)
    W = 0.5 * (W + W.T)

    # DD = sum(abs(W0));  % column sums
    DD = np.sum(np.abs(W0), axis=0)

    # MATLAB:
    # if length(unique(W(:))) == 2
    #     P = W;
    # else
    #     P = dominateset(double(abs(W)), min(K, length(W)-1)) .* sign(W);
    # end
    if np.unique(W).size == 2:
        P = W.copy()
    else:
        P = _dominateset(np.abs(W), min(K, W.shape[0] - 1)) * np.sign(W)

    # MATLAB:
    # P = P + (eye(length(P)) + diag(sum(abs(P'))));
    # note: sum(abs(P')) in MATLAB = row sums of abs(P)
    row_abs_sum = np.sum(np.abs(P), axis=1)
    P = P + np.eye(P.shape[0]) + np.diag(row_abs_sum)

    P = _transition_fields(P, eps=eps)

    # eig decomposition
    evals, evecs = np.linalg.eigh(P)  # P should be symmetric
    d = np.real(evals - eps)
    d = (1.0 - alpha) * d / (1.0 - alpha * np.power(d, order))

    W = evecs @ np.diag(np.real(d)) @ evecs.T

    # MATLAB:
    # W = (W .* (1-eye(length(W)))) ./ repmat(1-diag(W),1,length(W));
    diagW = np.diag(W).copy()
    denom = 1.0 - diagW
    denom = np.where(np.abs(denom) < eps, eps, denom)
    W = (W * (1.0 - np.eye(W.shape[0]))) / denom[:, None]

    # MATLAB:
    # D = sparse(1:length(DD),1:length(DD),(DD)); W = D*(W);
    W = DD[:, None] * W

    # clip negatives, symmetrize
    W[W < 0] = 0.0
    W = 0.5 * (W + W.T)

    W_out[np.ix_(keep, keep)] = W
    return W_out


def CLEAN(A_noisy,
          n_clusters=2,
          unlabeled_target_mean=0.1,
          consensus_thr=0.2,
          pos_thr=0.5
          ):
    from Model.model import train_edgenet
    from Model.clustering import build_partition_ensemble
    from Model.utils import (
        adj_to_edge_list,
        edge_list_to_csr,
        node2vec_embedding_from_adjacency,
        build_edge_score_bank,
        expand_edges,
        build_candidate_edges,
        build_edge_features,

    )
    n_nodes = A_noisy.shape[0]

    E_i, E_j = adj_to_edge_list(A_noisy)
    e_edges = len(E_i)
    print(f"[Graph] |E_noisy|={e_edges}")

    A_csr = edge_list_to_csr(n_nodes, E_i, E_j, undirected=True)

    # =============================
    # 2) EdgeNet-specific preparation
    # =============================
    partitions, used_algorithms = build_partition_ensemble(
        A_noisy=A_noisy,
        n_clusters=n_clusters,
        # mode='default'
    )

    emb = node2vec_embedding_from_adjacency(A_noisy)

    bank = build_edge_score_bank(A=A_csr, emb=emb)

    top_i, top_j = expand_edges(A_csr, bank)
    cand = build_candidate_edges(
        E_i=E_i, E_j=E_j,
        top_i=top_i, top_j=top_j,
        partitions=partitions,
        consensus_thr=consensus_thr
    )

    edge_feats = build_edge_features(cand=cand, bank=bank, pos_thr=pos_thr)

    all_i = np.triu_indices(n_nodes, k=1)[0].astype(np.int64)
    all_j = np.triu_indices(n_nodes, k=1)[1].astype(np.int64)
    train_mask = edge_feats.y >= 0

    W_pred, model, history = train_edgenet(
        X=edge_feats.X,
        y=edge_feats.y,
        train_mask=train_mask,
        i_idx=all_i,
        j_idx=all_j,
        n_nodes=n_nodes,
        unlabeled_target_mean=unlabeled_target_mean,
        # es_min_delta=1e-3
    )

    from Model.utils import connect_graph_with_tiny_edges
    W_pred = connect_graph_with_tiny_edges(W_pred)

    return W_pred

def noisy(A_noisy):
    return A_noisy