from __future__ import annotations
import numpy as np


# -----------------------------
# Data simulation (SBM)
# -----------------------------
def generate_sbm_with_noise(
    community_sizes=(300, 300, 300),
    p_in: float = 0.20,
    p_out: float = 0.10,
    noise_ratio: float = 0.0,  # ratio of perturbation
    noise_type: str = "add",  # "add" or "remove"
    seed: int = 64,
):
    """
    Generate a clean SBM graph A_clean (0/1) and
    a noisy graph A_noisy (0/1) by either adding or removing edges globally.

    noise_ratio: fraction of clean edges to perturb
                 if noise_type="add": number_added = floor(noise_ratio * #clean_edges)
                 if noise_type="remove": number_removed = floor(noise_ratio * #clean_edges)

    Returns:
      A_clean: [N,N] float32, symmetric 0/1, diag=0
      A_noisy: [N,N] float32, symmetric 0/1, diag=0
      y_true : [N]   int64
    """
    rng = np.random.RandomState(seed)

    # ----- labels -----
    N = sum(community_sizes)
    y_true = np.empty(N, dtype=np.int64)
    start = 0
    for k, size in enumerate(community_sizes):
        y_true[start:start + size] = k
        start += size

    # ----- clean SBM -----
    A = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if (y_true[i] == y_true[j]) else p_out
            if rng.rand() < p:
                A[i, j] = 1
                A[j, i] = 1
    np.fill_diagonal(A, 0)
    A_clean = A.copy()

    # ----- count clean edges on upper triangle -----
    A_triu = np.triu(A_clean, k=1)
    clean_edges = np.argwhere(A_triu == 1)
    num_clean = clean_edges.shape[0]

    # ----- apply noise: either add or remove, not both -----
    if noise_type == "remove":
        # remove edges (false negative)
        num_perturb = int(noise_ratio * num_clean)
        if num_perturb > 0 and num_clean > 0:
            num_perturb = min(num_perturb, num_clean)
            rm_idx = rng.choice(num_clean, size=num_perturb, replace=False)
            rm_edges = clean_edges[rm_idx]
            A_triu[rm_edges[:, 0], rm_edges[:, 1]] = 0
    elif noise_type == "add":
        # add edges (false positive)
        non_edges = np.argwhere(A_triu == 0)
        num_perturb = int(noise_ratio * num_clean)
        if num_perturb > 0 and non_edges.shape[0] > 0:
            num_perturb = min(num_perturb, non_edges.shape[0])
            add_idx = rng.choice(non_edges.shape[0], size=num_perturb, replace=False)
            add_edges = non_edges[add_idx]
            A_triu[add_edges[:, 0], add_edges[:, 1]] = 1
    else:
        raise ValueError("noise_type must be 'add' or 'remove'")

    # ----- symmetrize -----
    A_noisy = A_triu + A_triu.T
    A_noisy = 1 * (A_noisy > 0)
    np.fill_diagonal(A_noisy, 0)

    return A_clean, A_noisy, y_true


# -----------------------------
# Data simulation (Karate)
# -----------------------------
def load_karate_with_noise(
    noise_ratio: float = 0.3,  # fraction of clean edges to perturb
    noise_type: str = "add",   # "add" or "remove"
    seed: int = 64,
    verbose: bool = True,
    save: bool = True
):
    """
    Load Zachary's Karate Club graph and create a noisy version by either
    adding spurious edges or removing existing edges.

    Args:
        noise_ratio: proportion of clean edges used for perturbation.
                     If noise_type="add": #added ≈ noise_ratio * #clean_edges
                     If noise_type="remove": #removed ≈ noise_ratio * #clean_edges
        noise_type: "add" for false positives, "remove" for false negatives.
        seed: random seed for reproducibility.
        verbose: whether to print summary statistics.

    Returns:
        A_clean: [34,34] float32, symmetric 0/1
        A_noisy: [34,34] float32, symmetric 0/1
        y_true : [34]    int64 (club label mapped to {0,1})
    """
    import networkx as nx

    rng = np.random.default_rng(seed)

    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    assert N == 34

    # ----- Build clean adjacency matrix -----
    A_clean = np.zeros((N, N), dtype=np.float32)
    for u, v in G.edges():
        A_clean[u, v] = 1.0
        A_clean[v, u] = 1.0

    # ----- Extract ground-truth labels -----
    clubs = [G.nodes[i]["club"] for i in range(N)]
    unique_clubs = sorted(set(clubs))
    club2id = {c: idx for idx, c in enumerate(unique_clubs)}
    y = np.array([club2id[c] for c in clubs], dtype=np.int64)
    y[8] = 1  # fix label

    A_noisy = A_clean.copy()

    # ----- Get clean edges in upper triangle -----
    edges = np.argwhere(np.triu(A_clean, k=1) == 1)
    E_clean = edges.shape[0]

    # ----- Apply noise: add or remove -----
    if noise_type == "remove":
        num_perturb = int(noise_ratio * E_clean)
        if num_perturb > 0:
            num_perturb = min(num_perturb, E_clean)
            perm = rng.permutation(E_clean)
            remove_edges = edges[perm[:num_perturb]]
            for i, j in remove_edges:
                A_noisy[i, j] = 0.0
                A_noisy[j, i] = 0.0
    elif noise_type == "add":
        non_edges = np.argwhere(np.triu(1 - A_clean, k=1) == 1)
        num_perturb = int(noise_ratio * E_clean)
        if num_perturb > 0 and non_edges.shape[0] > 0:
            num_perturb = min(num_perturb, non_edges.shape[0])
            perm = rng.permutation(non_edges.shape[0])
            add_edges = non_edges[perm[:num_perturb]]
            for i, j in add_edges:
                A_noisy[i, j] = 1.0
                A_noisy[j, i] = 1.0
    else:
        raise ValueError("noise_type must be 'add' or 'remove'")

    if verbose:
        print(
            f"[Karate] N={N}, #edges_clean={E_clean}, #edges_noisy={int(np.triu(A_noisy,1).sum())}"
        )
        print(
            f"[Karate] noise_type={noise_type}, noise_ratio={noise_ratio}, num_perturb={num_perturb}"
        )

    if save:
        import os
        save_file = './Data/Karate/'
        os.makedirs(save_file, exist_ok=True)
        np.savez(
            save_file + "network_with_label.npz",
            Network=A_noisy,
            label=y
        )
    return A_clean, A_noisy, y


def add_noise_to_adjacency(
    A: np.ndarray,
    noise_ratio: float = 0.3,
    noise_type: str = "add",   # "add", "remove", "flip"
    seed: int = 42,
    keep_symmetry: bool = True,
    allow_self_loops: bool = False,
    verbose: bool = True,
):
    """
    Add noise to a graph adjacency matrix.

    Args:
        A: [N, N] adjacency matrix (0/1 or weighted)
        noise_ratio: fraction of edges used for perturbation
        noise_type:
            - "add": add edges (false positives)
            - "remove": remove edges (false negatives)
            - "flip": randomly add/remove edges
        seed: random seed
        keep_symmetry: enforce A[i,j] == A[j,i]
        allow_self_loops: whether to allow diagonal perturbation
        verbose: print stats

    Returns:
        A_noisy: perturbed adjacency matrix
    """
    rng = np.random.default_rng(seed)

    A = A.copy()
    N = A.shape[0]

    assert A.shape[0] == A.shape[1], "A must be square"

    # ----- upper triangle mask -----
    k = 0 if allow_self_loops else 1
    triu_mask = np.triu(np.ones_like(A), k=k)

    edges = np.argwhere((A > 0) & (triu_mask == 1))
    non_edges = np.argwhere((A == 0) & (triu_mask == 1))

    E = len(edges)
    num_perturb = int(noise_ratio * E)

    A_noisy = A.copy()

    # ----- REMOVE edges -----
    if noise_type == "remove":
        if E > 0:
            num_perturb = min(num_perturb, E)
            idx = rng.permutation(E)[:num_perturb]
            for i, j in edges[idx]:
                A_noisy[i, j] = 0
                if keep_symmetry:
                    A_noisy[j, i] = 0

    # ----- ADD edges -----
    elif noise_type == "add":
        if len(non_edges) > 0:
            num_perturb = min(num_perturb, len(non_edges))
            idx = rng.permutation(len(non_edges))[:num_perturb]
            for i, j in non_edges[idx]:
                A_noisy[i, j] = 1
                if keep_symmetry:
                    A_noisy[j, i] = 1

    # ----- FLIP edges (add + remove) -----
    elif noise_type == "flip":
        # half remove, half add
        num_remove = num_perturb // 2
        num_add = num_perturb - num_remove

        # remove
        if E > 0:
            num_remove = min(num_remove, E)
            idx = rng.permutation(E)[:num_remove]
            for i, j in edges[idx]:
                A_noisy[i, j] = 0
                if keep_symmetry:
                    A_noisy[j, i] = 0

        # add
        if len(non_edges) > 0:
            num_add = min(num_add, len(non_edges))
            idx = rng.permutation(len(non_edges))[:num_add]
            for i, j in non_edges[idx]:
                A_noisy[i, j] = 1
                if keep_symmetry:
                    A_noisy[j, i] = 1

    else:
        raise ValueError("noise_type must be 'add', 'remove', or 'flip'")

    if verbose:
        E_clean = int(np.triu(A, 1).sum())
        E_noisy = int(np.triu(A_noisy, 1).sum())
        print(f"[Graph] N={N}, edges_clean={E_clean}, edges_noisy={E_noisy}")
        print(f"[Graph] noise_type={noise_type}, noise_ratio={noise_ratio}, num_perturb={num_perturb}")

    return A_noisy