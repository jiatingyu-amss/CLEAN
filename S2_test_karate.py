from Model.clustering import build_partition_ensemble
from Model.data_generation import load_karate_with_noise
from Model.evaluation import *
from Model.model import train_edgenet
from Model.plot_utils import *
from Model.utils import *
SEED=64
set_seed(SEED)


# -----------------------------
# Global config
# -----------------------------
DATASET = "karate"
noise_ratio = 0
noise_type = 'add'
save_file = f'./Results/{DATASET}/'
os.makedirs(save_file, exist_ok=True)


# -----------------------------
# 1) Data
# -----------------------------
A_clean, A_noisy, y_true = load_karate_with_noise(
    noise_type=noise_type,
    noise_ratio=noise_ratio,
    save=True
    )

n_nodes = A_noisy.shape[0]
n_clusters = int(len(np.unique(y_true)))
print(f"[Data] DATASET={DATASET}, N={n_nodes}, n_clusters={n_clusters}")

# Build edge list from adjacency
E_i, E_j = adj_to_edge_list(A_noisy)
e_edges = len(E_i)
print(f"[Graph] |E_noisy|={e_edges}")

# Build CSR adjacency
A_csr = edge_list_to_csr(n_nodes, E_i, E_j, undirected=True)


# --------------------------------------
# 2) Pseudo-label ensemble partitions
# --------------------------------------
partitions, used_algorithms = build_partition_ensemble(
    A_noisy=A_noisy,
    n_clusters=n_clusters,
    random_state=SEED,
    mode='default'
)


# -----------------------------------------
# 3) Node embedding (for edge expansion)
# -----------------------------------------
emb = node2vec_embedding_from_adjacency(A_noisy)


# --------------------------------------
# 4) Candidate Edge and Edge features
# --------------------------------------
bank = build_edge_score_bank(A=A_csr, emb=emb)

top_i, top_j = expand_edges(A_csr, bank)
cand = build_candidate_edges(
    E_i=E_i, E_j=E_j,
    top_i=top_i, top_j=top_j,
    partitions=partitions
)
print(f"[Cand] |E_cand|={len(cand.i)})")

edge_feats = build_edge_features(cand=cand, bank=bank)
X = edge_feats.X
y = edge_feats.y

num_total = len(y)
num_pos = np.sum(y == 1)
num_neg = np.sum(y == 0)
num_unl = np.sum(y == -1)
print(f"[Labels] total={num_total}")
print(f"[Labels] pos (1) : {num_pos} ({num_pos/num_total:.3f})")
print(f"[Labels] neg (0) : {num_neg} ({num_neg/num_total:.3f})")
print(f"[Labels] unl (-1): {num_unl} ({num_unl/num_total:.3f})")
print(f"[Feats] X shape={X.shape}")



# -----------------------------
# 5) Train EdgeNet
# -----------------------------
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
    es_min_delta=1e-3
)

# -----------------------------
# 6) Evaluation + visualization
# -----------------------------
plot_training_history(history, save_file)

plot_mats = {"noisy": A_noisy, "denoised": W_pred}
visualize_matrices(
    matrices=plot_mats, y_true=y_true,
    save_path=save_file+'visualize_matrices.svg'
)
visualize_network_dict(
    graphs_dict=plot_mats,
    y_true=y_true,
    save_dir=os.path.join(save_file, "network")
)

snr = snr_cohen_d(W_pred, y_true)
print(f"[Eval] snr = {snr:.4f}")

results = evaluate_downstream_clustering(
    A_noisy, W_pred, y_true,
    algorithms=used_algorithms,
    save_path=save_file,
    random_state=SEED
)
plot_downstream_clustering_bars(results, save_file)