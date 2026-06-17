from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import copy


class EdgeNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        residual_scale: float = 0.3,
    ):
        super().__init__()

        self.residual_scale = float(residual_scale)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Residual edge weighting:
            s_base = x[:, 0]
            delta = residual_scale * tanh(MLP(x))
            w = clamp(s_base + delta, 0, 1)

        Returns:
            w      : final edge weights in [0, 1]
            delta  : residual correction
            s_base : baseline consensus score
        """
        s_base = x[:, 0]
        raw_delta = self.mlp(x).squeeze(-1)
        delta = self.residual_scale * torch.tanh(raw_delta)
        w = torch.clamp(s_base + delta, min=0.0, max=1.0)
        return w, delta, s_base


class EarlyStopper:
    """
    Early stop on a scalar metric (here: total loss).
    - mode='min': smaller is better
    - patience: stop after N epochs w/o improvement
    - min_delta: required improvement margin
    - warmup: ignore early-stop checks for first warmup epochs (but still track best)
    - ema_alpha: optional smoothing for noisy loss curves (None to disable)
    """
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
        warmup: int = 10,
        ema_alpha: float | None = 0.2,
        mode: str = "min",
    ):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup = int(warmup)
        self.ema_alpha = ema_alpha
        self.mode = mode

        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = -1
        self.bad_count = 0

        self._ema = None
        self.best_state = None  # deepcopy(state_dict)

    def _is_improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)

    def step(self, value: float, epoch: int, model: torch.nn.Module) -> tuple[bool, float]:
        v = float(value)

        # optional EMA smoothing
        if self.ema_alpha is not None:
            if self._ema is None:
                self._ema = v
            else:
                a = float(self.ema_alpha)
                self._ema = a * v + (1.0 - a) * self._ema
            v_used = float(self._ema)
        else:
            v_used = v

        # always keep best (even in warmup)
        if self._is_improved(v_used):
            self.best = v_used
            self.best_epoch = int(epoch)
            self.bad_count = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            if epoch > self.warmup:
                self.bad_count += 1

        should_stop = (epoch > self.warmup) and (self.bad_count >= self.patience)
        return should_stop, v_used


def train_edgenet(
    X: np.ndarray,             # [E, D], first column must be s_ij
    y: np.ndarray,             # [E], supervised labels in {0,1}, unlabeled = -1
    train_mask: np.ndarray,    # [E], supervised edges mask
    i_idx: np.ndarray,         # [E]
    j_idx: np.ndarray,         # [E]
    n_nodes: int,              # number of nodes in graph
    num_epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    grad_clip: float = 1.0,
    debug_every: int = 100,
    early_stop: bool = True,
    es_patience: int = 100,
    es_min_delta: float = 1e-5,
    es_warmup: int = 10,
    es_ema_alpha: Optional[float] = 0.2,
    lambda_unlabeled: float = 0.1,
    unlabeled_target_mean: float = 0.1,
    residual_scale: float = 0.3,
    hidden_dim: int = 32,
    dropout: float = 0.1,
) -> Tuple[np.ndarray, "EdgeNet", Dict[str, list]]:
    """
    Train residual edge denoising network.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).float().to(device)
    mask_t = torch.from_numpy(train_mask.astype(bool)).to(device)

    # Unlabeled edges are marked by y = -1
    unl_mask_t = (yt < 0)

    num_supervised = int(mask_t.sum().item())
    num_unlabeled = int(unl_mask_t.sum().item())

    model = EdgeNet(
        in_dim=X.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual_scale=residual_scale,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCELoss(reduction="none")

    history: Dict[str, list] = {
        "loss": [],
        "loss_sup": [],
        "loss_unl": [],
        "mean_w": [],
        "sup_mean_w": [],
        "unl_mean_w": [],
        "mean_abs_delta": [],
        "num_supervised": [],
        "num_positive": [],
        "num_negative": [],
        "num_unlabeled": [],
        "loss_used_for_es": [],
        "best_epoch": [],
    }

    if num_supervised == 0:
        history["num_supervised"].append(0)
        history["num_positive"].append(0)
        history["num_negative"].append(0)
        history["num_unlabeled"].append(num_unlabeled)

        W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        if debug_every > 0:
            print("[Train] No supervised samples. Return all-zero W_pred.")
        return W, model, history

    stopper = None
    if early_stop:
        stopper = EarlyStopper(
            patience=es_patience,
            min_delta=es_min_delta,
            warmup=es_warmup,
            ema_alpha=es_ema_alpha,
            mode="min",
        )

    for ep in range(1, num_epochs + 1):
        model.train()

        w, delta, s_base = model(Xt)

        # Supervised BCE on labeled edges
        sup_w = w[mask_t]
        sup_targets = yt[mask_t]
        loss_sup = bce(sup_w, sup_targets).mean()

        # Unlabeled mean regularization
        if num_unlabeled > 0:
            unl_w = w[unl_mask_t]
            loss_unl = (unl_w.mean() - float(unlabeled_target_mean)) ** 2
        else:
            loss_unl = torch.zeros((), device=device, dtype=torch.float32)

        loss_total = loss_sup + float(lambda_unlabeled) * loss_unl

        opt.zero_grad(set_to_none=True)
        loss_total.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        opt.step()

        w_np = w.detach().cpu().numpy().astype(np.float32)
        delta_np = delta.detach().cpu().numpy().astype(np.float32)
        sup_w_np = w[mask_t].detach().cpu().numpy().astype(np.float32)
        unl_w_np = w[unl_mask_t].detach().cpu().numpy().astype(np.float32) if num_unlabeled > 0 else None

        loss_total_val = float(loss_total.detach().cpu())
        loss_sup_val = float(loss_sup.detach().cpu())
        loss_unl_val = float(loss_unl.detach().cpu())

        history["loss"].append(loss_total_val)
        history["loss_sup"].append(loss_sup_val)
        history["loss_unl"].append(loss_unl_val)
        history["mean_w"].append(float(w_np.mean()))
        history["sup_mean_w"].append(float(sup_w_np.mean()))
        history["unl_mean_w"].append(float(unl_w_np.mean()) if unl_w_np is not None else 0.0)
        history["mean_abs_delta"].append(float(np.mean(np.abs(delta_np))))
        history["num_supervised"].append(num_supervised)
        history["num_positive"].append(int((yt[mask_t] > 0.5).sum().item()))
        history["num_negative"].append(int((yt[mask_t] <= 0.5).sum().item()))
        history["num_unlabeled"].append(num_unlabeled)

        if debug_every > 0 and (ep % debug_every == 0 or ep == 1):
            msg = (
                f"[Train] ep={ep:03d} "
                f"w_mean={w_np.mean():.4f} "
                f"sup_w_mean={sup_w_np.mean():.4f} "
                f"|delta|_mean={np.mean(np.abs(delta_np)):.4f} "
                f"L={loss_total_val:.4f} "
                f"L_sup={loss_sup_val:.4f} "
                f"L_unl={loss_unl_val:.4f} "
            )
            if unl_w_np is not None:
                msg += f" unl_w_mean={unl_w_np.mean():.4f}"
            print(msg)

        if stopper is not None:
            should_stop, loss_used = stopper.step(value=loss_total_val, epoch=ep, model=model)
            history["loss_used_for_es"].append(float(loss_used))
            history["best_epoch"].append(int(stopper.best_epoch))

            if should_stop:
                if debug_every > 0:
                    print(
                        f"[EarlyStop] stop at ep={ep:03d} "
                        f"(best_ep={stopper.best_epoch:03d}, best_loss={stopper.best:.6f})"
                    )
                break
        else:
            history["loss_used_for_es"].append(loss_total_val)
            history["best_epoch"].append(ep)

    if stopper is not None and stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    model.eval()
    with torch.no_grad():
        w_final, _, _ = model(Xt)
        w_final = w_final.detach().cpu().numpy().astype(np.float32)

    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    ii = i_idx.astype(np.int64)
    jj = j_idx.astype(np.int64)
    W[ii, jj] = np.maximum(W[ii, jj], w_final)
    W[jj, ii] = np.maximum(W[jj, ii], w_final)
    np.fill_diagonal(W, 0.0)
    W[W < 1e-3] = 0.0

    return W, model, history