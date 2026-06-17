from Model.clustering import build_partition_ensemble
from Model.data_generation import generate_sbm_with_noise
from Model.model import train_edgenet
from Model.plot_utils import *
from Model.utils import *
SEED = 64
set_seed(SEED)


# -----------------------------
# Global config
# -----------------------------
DATASET = "sbm"
save_file = f'./Results/{DATASET}/'
os.makedirs(save_file, exist_ok=True)

community_sizes = (12, 15, 10, 8)

# # ---- easy ----
# p_in = 0.7
# p_out = 0.1
# noise_ratio = 0.1
# noise_type='add'
# consensus_thr = 0.2
# pos_thr = 0.5


# # ---- medium ----
p_in = 0.5
p_out = 0.1
noise_ratio = 0.1
noise_type='add'
pos_thr = 0.6


# 1) Data
A_clean, A_noisy, y_true = generate_sbm_with_noise(
    community_sizes=community_sizes,
    p_in=p_in,
    p_out=p_out,
    noise_ratio=noise_ratio,
    noise_type=noise_type,
    seed=SEED
)

n_nodes = A_noisy.shape[0]
n_clusters = int(len(np.unique(y_true)))
print(f"[Data] DATASET={DATASET}, N={n_nodes}, n_clusters={n_clusters}")

E_i, E_j = adj_to_edge_list(A_noisy)
e_edges = len(E_i)
print(f"[Graph] |E_noisy|={e_edges}")

A_csr = edge_list_to_csr(n_nodes, E_i, E_j, undirected=True)


# --------------------------------------
# 2) Pseudo-label ensemble partitions
# --------------------------------------
partitions, used_algorithms = build_partition_ensemble(
    A_noisy=A_noisy,
    n_clusters=n_clusters,
    mode="default"
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

edge_feats = build_edge_features(cand=cand, bank=bank, pos_thr=pos_thr)
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
train_mask = y >= 0

W_pred, model, history = train_edgenet(
    X=X,
    y=y,
    train_mask=train_mask,
    i_idx=all_i,
    j_idx=all_j,
    n_nodes=n_nodes
)


# -----------------------------
# 6) Evaluation + visualization
# -----------------------------
plot_training_history(history, save_file)

visualize_matrices({"noisy": A_noisy}, y_true=y_true, save_path=save_file+'noisy_matrices.svg')
visualize_matrices({"denoised": W_pred}, y_true=y_true, save_path=save_file+'denoised_matrices.svg')

visualize_network_dict(
    graphs_dict={"denoised": W_pred, "noisy": A_noisy},
    y_true=y_true,
    layout='community',
    save_dir=os.path.join(save_file, "network")
)

# Clean network
W_pred_cut = W_pred.copy()
W_pred_cut[W_pred_cut<0.6]=0

visualize_matrices({"denoised (clean)": W_pred_cut}, y_true=y_true, save_path=save_file+'denoised_clean_matrices.svg')
visualize_network_dict(
    graphs_dict={"denoised (clean)": W_pred_cut},
    y_true=y_true,
    layout='community',
    save_dir=os.path.join(save_file, "network")
)