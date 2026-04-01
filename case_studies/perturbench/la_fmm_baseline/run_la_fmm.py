"""
LA+FMM: Latent Additive + Finite Mixture Model for PerturBench Srivatsan20.

Combines the best of both worlds:
- LA baseline: Excellent mean prediction via neural network
- FMM baseline: Good distributional quality via GMM sampling

Pipeline:
1. Train LA model (same architecture as la_reproduced)
2. Compute per-cell effects and fit PCA (from fmm_baseline)
3. For each (pert, target_ct) prediction:
   - Use trained LA model to predict mean effect
   - Pool PCA-projected effects from source CTs
   - Fit GMM on pooled effects with CT-specific mixing proportions
   - Recenter GMM on LA prediction (key hybrid step)
   - Sample from recentered GMM to generate heterogeneous predictions
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import logging
import warnings
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

from perturbench.analysis.benchmarks.evaluator import Evaluator
from perturbench.modelcore.nn.mlp import MLP

import scanpy as sc
sc.settings.verbosity = 2

# ---- Paths & Task ----
DATA_CACHE = 'eval/perturbench_eval/perturbench_files/data'
TASK = 'srivatsan20-transfer'
OUT_DIR = 'eval/perturbench_eval/la_fmm_baseline'

# ---- LA hyperparameters (EXACT from run_la_reproduced.py) ----
ENCODER_WIDTH = 5376
LATENT_DIM = 192
N_LAYERS = 2
DROPOUT = 0.3
LR = 3.4e-5
WD = 6.1e-11
BATCH_SIZE = 4000
MAX_EPOCHS = 500

# ---- LA variant toggles (EXACT configuration for baseline) ----
INJECT_COVARIATES = True   # True = concat ct_onehot to encoder/decoder (published LA)
USE_CTRL_MEAN = False      # Per-cell control sampling during training
PREDICT_PERCELL = True     # Average over all control cells at inference

# ---- Disable auxiliary losses (vanilla LA) ----
LAMBDA_CORAL = 0.0
LAMBDA_CONTRASTIVE = 0.0

# ---- Training ----
PATIENCE_LR = 15
PATIENCE_EARLY = 50
SEED = 245

# ---- GMM settings (from fmm_baseline) ----
N_PCA = 200
K_VALUES = [2, 3, 5, 7, 10]
MIN_CELLS_FOR_GMM = 20
USE_RESIDUAL = True  # Preserve LA signal outside PCA subspace

# ---- Pipeline control ----
REUSE_LA_CHECKPOINT = True
# Load checkpoints from la_reproduced (trained with matching config)
CHECKPOINT_DIR = 'eval/perturbench_eval/la_reproduced/results/checkpoints'
CHECKPOINT_NAME = None  # Set dynamically from SEED below
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else (
    'mps' if torch.backends.mps.is_available() else 'cpu')

# --- CLI seed override ---
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--seed', type=int, default=None, help='Override SEED')
_args, _ = _parser.parse_known_args()
if _args.seed is not None:
    SEED = _args.seed
CHECKPOINT_NAME = f'la_model_seed{SEED}.pt'

# --- Logging setup ---
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(OUT_DIR, 'results', f'run_{RUN_TIMESTAMP}_seed{SEED}')
os.makedirs(run_dir, exist_ok=True)

logger = logging.getLogger('la_fmm')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(os.path.join(run_dir, 'run.log'), mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)


# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def to_dense(X):
    """Convert sparse matrix to dense array."""
    if issparse(X):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def compute_cell_effects(train_adata, pert_col, ct_col, ctrl):
    """Compute per-cell effects and control means per cell type.

    COPIED FROM: run_fmm_baseline.py lines 63-88
    """
    obs = train_adata.obs
    ctrl_means = {}
    for ct in obs[ct_col].unique():
        mask = (obs[pert_col] == ctrl) & (obs[ct_col] == ct)
        if mask.sum() > 0:
            ctrl_means[ct] = to_dense(train_adata[mask].X).mean(axis=0)

    cell_effects = {}
    mean_effects = {}
    for ct in obs[ct_col].unique():
        if ct not in ctrl_means:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            mask = ct_mask & (obs[pert_col] == pert)
            if mask.sum() > 0:
                cells = to_dense(train_adata[mask].X)
                effects = cells - ctrl_means[ct]
                cell_effects[(pert, ct)] = effects
                mean_effects[(pert, ct)] = effects.mean(axis=0)

    return cell_effects, mean_effects, ctrl_means


def compute_pert_response_similarity(mean_effects):
    """Similarity between cell types based on perturbation response correlation.

    COPIED FROM: run_fmm_baseline.py lines 91-117

    For each CT pair, find shared perturbations and compute average cosine
    similarity of their mean effect vectors across those shared drugs.
    """
    ct_perts = {}
    for (pert, ct) in mean_effects:
        ct_perts.setdefault(ct, set()).add(pert)

    cts = list(ct_perts.keys())
    sims = {}
    for ct_a in cts:
        for ct_b in cts:
            if ct_a == ct_b:
                continue
            shared = ct_perts[ct_a] & ct_perts[ct_b]
            if not shared:
                sims[(ct_a, ct_b)] = 0.5
                continue
            cos_sims = []
            for p in shared:
                a, b = mean_effects[(p, ct_a)], mean_effects[(p, ct_b)]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                cos_sims.append(cos)
            sims[(ct_a, ct_b)] = np.mean(cos_sims)
    return sims


def _add_noise(pred_X, mask, predicted_expr, rng):
    """Add small noise to predictions.

    COPIED FROM: run_fmm_baseline.py lines 171-177
    """
    n_cells = mask.sum()
    noise = rng.normal(0, 1e-6, size=(n_cells, predicted_expr.shape[-1] if predicted_expr.ndim > 1 else len(predicted_expr)))
    if predicted_expr.ndim == 1:
        pred_X[mask] = predicted_expr + noise
    else:
        pred_X[mask] = predicted_expr + noise


# ===============================================================================
# DATASET & MODEL (EXACT FROM run_la_reproduced.py)
# ===============================================================================

class LADataset(Dataset):
    """Per-cell dataset for LA training. Returns control expression for the cell's
    cell type, perturbation/cell-type one-hots, indices, and target expression.

    COPIED FROM: run_la_reproduced.py lines 94-131
    """

    def __init__(self, expr, ct_indices, pert_indices,
                 pert_onehots, ct_onehots, ctrl_means,
                 ctrl_expr_all=None, ctrl_indices_by_ct=None):
        self.expr = expr                 # (N, n_genes) float32
        self.ct_indices = ct_indices     # (N,) int64
        self.pert_indices = pert_indices # (N,) int64
        self.pert_onehots = pert_onehots # (N, n_perts) float32
        self.ct_onehots = ct_onehots     # (N, n_cts) float32
        self.ctrl_means = ctrl_means     # (n_cts, n_genes) float32
        # For per-cell control sampling (published LA behavior)
        self.ctrl_expr_all = ctrl_expr_all       # (n_ctrl, n_genes) or None
        self.ctrl_indices_by_ct = ctrl_indices_by_ct  # {ct_idx: np.array} or None

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, idx):
        ct_idx = self.ct_indices[idx]
        if self.ctrl_expr_all is not None:
            # Per-cell sampling: randomly pick one control cell from same CT
            ct_ctrl_idxs = self.ctrl_indices_by_ct[ct_idx]
            sampled = ct_ctrl_idxs[np.random.randint(len(ct_ctrl_idxs))]
            ctrl = torch.from_numpy(self.ctrl_expr_all[sampled])
        else:
            # CT mean
            ctrl = torch.from_numpy(self.ctrl_means[ct_idx])
        return (
            ctrl,
            torch.from_numpy(self.pert_onehots[idx]),
            torch.from_numpy(self.ct_onehots[idx]),
            torch.tensor(ct_idx, dtype=torch.long),
            torch.tensor(self.pert_indices[idx], dtype=torch.long),
            torch.from_numpy(self.expr[idx]),
        )


class LatentAdditiveModel(nn.Module):
    """LA backbone with exposed latent representations for auxiliary losses.

    COPIED FROM: run_la_reproduced.py lines 136-159
    """

    def __init__(self, n_genes, n_perts, n_cts, inject_covariates=INJECT_COVARIATES):
        super().__init__()
        self.inject_covariates = inject_covariates
        enc_in = n_genes + n_cts if inject_covariates else n_genes
        dec_in = LATENT_DIM + n_cts if inject_covariates else LATENT_DIM
        self.gene_encoder = MLP(enc_in, ENCODER_WIDTH, LATENT_DIM, N_LAYERS, DROPOUT)
        self.pert_encoder = MLP(
            n_perts, ENCODER_WIDTH, LATENT_DIM, N_LAYERS, DROPOUT)
        self.decoder = MLP(dec_in, ENCODER_WIDTH, n_genes, N_LAYERS, DROPOUT)
        # Projection head for contrastive loss (not used in vanilla LA)
        self.effect_proj = nn.Sequential(
            nn.Linear(n_genes, 256), nn.ReLU(), nn.Linear(256, 16))

    def forward(self, ctrl_expr, pert_onehot, ct_onehot):
        enc_input = torch.cat([ctrl_expr, ct_onehot], dim=1) if self.inject_covariates else ctrl_expr
        latent_ctrl = self.gene_encoder(enc_input)
        latent_pert = self.pert_encoder(pert_onehot)
        latent_perturbed = latent_ctrl + latent_pert
        dec_input = torch.cat([latent_perturbed, ct_onehot], dim=1) if self.inject_covariates else latent_perturbed
        pred = F.softplus(self.decoder(dec_input))
        return pred, latent_ctrl, latent_perturbed


# ===============================================================================
# LA MODEL TRAINING
# ===============================================================================

def build_dataset(adata, pert_col, ct_col, ctrl, all_perts, all_cts, pert2idx, ct2idx, n_genes, n_cts, split_name):
    """Build LADataset from an adata split (train or val).

    ADAPTED FROM: run_la_reproduced.py lines 354-388
    """
    obs = adata.obs
    # Control means per cell type (always needed for prediction)
    cmeans = np.zeros((n_cts, n_genes), dtype=np.float32)
    ctrl_expr_all = None
    ctrl_indices_by_ct = None
    for ct in all_cts:
        mask = (obs[pert_col] == ctrl) & (obs[ct_col] == ct)
        if mask.sum() > 0:
            cmeans[ct2idx[ct]] = to_dense(adata[mask].X).mean(axis=0)
    # Per-cell control sampling data (published LA behavior)
    if not USE_CTRL_MEAN:
        ctrl_mask = obs[pert_col] == ctrl
        ctrl_expr_all = to_dense(adata[ctrl_mask].X)
        ctrl_ct_labels = np.array([ct2idx[c] for c in obs.loc[ctrl_mask, ct_col]])
        ctrl_indices_by_ct = {}
        for ct_i in range(n_cts):
            idxs = np.where(ctrl_ct_labels == ct_i)[0]
            if len(idxs) > 0:
                ctrl_indices_by_ct[ct_i] = idxs
    # Filter to perturbed cells only
    pmask = obs[pert_col] != ctrl
    padata = adata[pmask]
    pobs = padata.obs
    expr = to_dense(padata.X)
    ct_idx = np.array([ct2idx[c] for c in pobs[ct_col]], dtype=np.int64)
    pert_idx = np.array([pert2idx[p] for p in pobs[pert_col]], dtype=np.int64)
    poh = np.zeros((len(expr), len(all_perts)), dtype=np.float32)
    poh[np.arange(len(expr)), pert_idx] = 1.0
    coh = np.zeros((len(expr), n_cts), dtype=np.float32)
    coh[np.arange(len(expr)), ct_idx] = 1.0
    logger.info(f"  {split_name}: {len(expr)} perturbed cells")
    return LADataset(expr, ct_idx, pert_idx, poh, coh, cmeans,
                     ctrl_expr_all, ctrl_indices_by_ct), cmeans


def train_la_model(train_adata, val_adata, pert_col, ct_col, ctrl,
                   pert2idx, ct2idx, all_perts, all_cts, n_genes, n_perts, n_cts, device):
    """Train LA model and return checkpoint.

    ADAPTED FROM: run_la_reproduced.py lines 354-535 (training loop)
    """
    logger.info("\nTraining LA model...")

    # Build datasets
    train_ds, ctrl_means_train = build_dataset(
        train_adata, pert_col, ct_col, ctrl, all_perts, all_cts,
        pert2idx, ct2idx, n_genes, n_cts, "train")
    val_ds, ctrl_means_val = build_dataset(
        val_adata, pert_col, ct_col, ctrl, all_perts, all_cts,
        pert2idx, ct2idx, n_genes, n_cts, "val")

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    model = LatentAdditiveModel(n_genes, n_perts, n_cts).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    # Optimizer (exclude projection head if contrastive loss is disabled)
    main_params = [p for n, p in model.named_parameters()
                   if not n.startswith('effect_proj')]
    optimizer = torch.optim.Adam(main_params, lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PATIENCE_LR, factor=0.2)

    # Training loop
    logger.info(f"  Training (max {MAX_EPOCHS} epochs, early stop patience={PATIENCE_EARLY})...")
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(MAX_EPOCHS):
        # -- Train --
        model.train()
        train_mse_losses = []

        for batch in train_dl:
            ctrl_expr, pert_oh, ct_oh, ct_idx, pert_idx, target = \
                [b.to(device) for b in batch]

            pred, latent_ctrl, latent_perturbed = model(
                ctrl_expr, pert_oh, ct_oh)

            mse = F.mse_loss(pred, target)
            loss = mse  # Vanilla LA: only MSE loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse_losses.append(mse.item())

        train_mse = np.mean(train_mse_losses)

        # -- Validate --
        model.eval()
        val_mse_losses = []
        with torch.no_grad():
            for batch in val_dl:
                ctrl_expr, pert_oh, ct_oh, ct_idx, pert_idx, target = \
                    [b.to(device) for b in batch]
                pred, latent_ctrl, latent_perturbed = model(
                    ctrl_expr, pert_oh, ct_oh)

                mse = F.mse_loss(pred, target)
                val_mse_losses.append(mse.item())

        val_mse = np.mean(val_mse_losses)
        scheduler.step(val_mse)

        if epoch % 10 == 0 or val_mse < best_val_loss:
            logger.info(f"  Epoch {epoch:3d} | "
                       f"mse={train_mse:.4f}/{val_mse:.4f} | "
                       f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone()
                         for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE_EARLY:
                logger.info(f"  Early stopping at epoch {epoch}, "
                           f"best_val_mse={best_val_loss:.4f}")
                break

    logger.info(f"  Training done. Best val_mse={best_val_loss:.4f}")

    # Build checkpoint
    checkpoint = {
        'model_state_dict': best_state,
        'n_genes': n_genes,
        'n_perts': n_perts,
        'n_cts': n_cts,
        'pert2idx': pert2idx,
        'ct2idx': ct2idx,
        'val_mse': best_val_loss,
        'hyperparams': {
            'encoder_width': ENCODER_WIDTH,
            'latent_dim': LATENT_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT,
            'inject_covariates': INJECT_COVARIATES,
        }
    }

    return checkpoint, best_state


def load_or_train_la_model(train_adata, val_adata, pert_col, ct_col, ctrl,
                            pert2idx, ct2idx, all_perts, all_cts, n_genes, n_perts, n_cts, device):
    """Load LA model checkpoint or train from scratch."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    if REUSE_LA_CHECKPOINT and os.path.exists(checkpoint_path):
        logger.info(f"\nLoading LA model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"  Loaded: val_mse={checkpoint['val_mse']:.4f}")

        # Validate checkpoint integrity
        assert checkpoint['n_genes'] == n_genes
        assert checkpoint['n_perts'] == n_perts
        assert checkpoint['n_cts'] == n_cts

        best_state = checkpoint['model_state_dict']
    else:
        checkpoint, best_state = train_la_model(
            train_adata, val_adata, pert_col, ct_col, ctrl,
            pert2idx, ct2idx, all_perts, all_cts, n_genes, n_perts, n_cts, device)

        # Save checkpoint with timestamp
        save_name = f'la_model_{RUN_TIMESTAMP}.pt'
        save_path = os.path.join(CHECKPOINT_DIR, save_name)
        torch.save(checkpoint, save_path)
        logger.info(f"  Saved checkpoint to {save_path}")

    # Load model
    model = LatentAdditiveModel(n_genes, n_perts, n_cts).to(device)
    model.load_state_dict(best_state)
    model.eval()

    return model, checkpoint


# ===============================================================================
# LA PREDICTION HELPER
# ===============================================================================

def predict_with_la_model_percell(model, val_ctrl_cells, pert_onehot, ct_onehot,
                                   n_genes, device):
    """Generate LA prediction using PREDICT_PERCELL approach.

    Passes ALL control cells through model, averages predictions.
    This matches the exact inference mode from run_la_reproduced.py lines 545-588.

    Args:
        model: Trained LatentAdditiveModel
        val_ctrl_cells: (n_ctrl, n_genes) all control cells for target CT
        pert_onehot: (n_perts,) one-hot perturbation vector
        ct_onehot: (n_cts,) one-hot cell type vector

    Returns:
        predicted_effect: (n_genes,) predicted perturbation effect
    """
    model.eval()
    n_ctrl = len(val_ctrl_cells)

    # Batch all control cells
    ct_oh_batch = np.tile(ct_onehot, (n_ctrl, 1))
    pert_oh_batch = np.tile(pert_onehot, (n_ctrl, 1))

    ctrl_t = torch.tensor(val_ctrl_cells, dtype=torch.float32).to(device)
    ct_oh_t = torch.tensor(ct_oh_batch, dtype=torch.float32).to(device)
    pert_oh_t = torch.tensor(pert_oh_batch, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Pass ALL control cells through model
        pred_all, _, _ = model(ctrl_t, pert_oh_t, ct_oh_t)  # (n_ctrl, n_genes)
        # Average predictions: E[f(x)]
        pred_mean = pred_all.cpu().numpy().mean(axis=0)  # (n_genes,)

    # Compute effect: prediction - control mean
    ctrl_mean = val_ctrl_cells.mean(axis=0)
    predicted_effect = pred_mean - ctrl_mean

    return predicted_effect


# ===============================================================================
# LA+FMM HYBRID PREDICTION
# ===============================================================================

def build_la_fmm_prediction(val_ref, cell_effects, ctrl_means_val,
                            pca, projected_effects, pert_sims,
                            la_model, la_checkpoint, val_ctrl_data,
                            K, pert_col, ct_col, ctrl, device):
    """LA+FMM: LA mean prediction + FMM GMM distributional structure.

    BASED ON: run_fmm_baseline.py lines 402-482 (build_linear_fmm_prediction)
    but replacing linear transfer with LA model inference.
    """
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)
    min_cells_for_gmm = MIN_CELLS_FOR_GMM

    n_perts = la_checkpoint['n_perts']
    n_cts = la_checkpoint['n_cts']
    pert2idx = la_checkpoint['pert2idx']
    ct2idx = la_checkpoint['ct2idx']
    n_genes = la_checkpoint['n_genes']

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        ct_idx = ct2idx[ct]

        # Get all control cells for this CT in val split
        val_ctrl_cells = val_ctrl_data[ct]  # (n_ctrl, n_genes)
        if len(val_ctrl_cells) == 0:
            continue

        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue

            # 1. Pool training effects from source CTs (same as LFMM)
            source_cts = [sct for sct in ctrl_means_val
                         if sct != ct and (pert, sct) in projected_effects]
            if not source_cts:
                continue

            pooled_pca_arr = np.concatenate([projected_effects[(pert, sct)]
                                           for sct in source_cts], axis=0)
            n_pooled = len(pooled_pca_arr)
            mask = (ct_mask & (obs[pert_col] == pert)).values
            n_cells = mask.sum()

            # 2. LA model mean prediction (REPLACES linear transfer)
            pert_oh = np.zeros(n_perts, dtype=np.float32)
            pert_oh[pert2idx[pert]] = 1.0
            ct_oh = np.zeros(n_cts, dtype=np.float32)
            ct_oh[ct_idx] = 1.0

            # Use EXACT LA inference mode (PREDICT_PERCELL)
            predicted_effect = predict_with_la_model_percell(
                la_model, val_ctrl_cells, pert_oh, ct_oh, n_genes, device)

            # 3. Fallback to LA-only if too few cells for GMM
            if n_pooled < min_cells_for_gmm:
                predicted_expr = ctrl_means_val[ct] + predicted_effect
                _add_noise(pred_X, mask, predicted_expr, rng)
                continue

            # 4. Fit GMM on pooled training effects (same as LFMM)
            actual_k = min(K, n_pooled // 5)
            actual_k = max(actual_k, 1)

            gmm = GaussianMixture(n_components=actual_k, covariance_type='diag',
                                 random_state=42, max_iter=200)
            gmm.fit(pooled_pca_arr)

            # 5. Compute FMM proportions (same as LFMM)
            ct_weights = np.array([pert_sims.get((ct, sct), 0.5) for sct in source_cts])
            ct_weights /= ct_weights.sum()

            pi_per_ct = []
            for sct in source_cts:
                assignments = gmm.predict(projected_effects[(pert, sct)])
                pi = np.bincount(assignments, minlength=actual_k) / len(assignments)
                pi_per_ct.append(pi)

            target_pi = sum(w * pi for w, pi in zip(ct_weights, pi_per_ct))
            target_pi /= target_pi.sum()
            gmm.weights_ = target_pi

            # 6. Recenter GMM on LA prediction (same as LFMM)
            mu_gmm = np.average(gmm.means_, axis=0, weights=target_pi)
            mu_la = pca.transform(predicted_effect.reshape(1, -1)).ravel()
            gmm.means_ += (mu_la - mu_gmm)  # Shift all component means

            # 7. Sample from recentered GMM
            sampled_pca, _ = gmm.sample(n_cells)
            sampled_effects = pca.inverse_transform(sampled_pca.astype(np.float32))

            if USE_RESIDUAL:
                # Preserve LA signal outside PCA subspace:
                # residual = predicted_effect - pca_roundtrip(predicted_effect)
                # final = sampled_pca_effects + residual
                pca_roundtrip = pca.inverse_transform(mu_la.reshape(1, -1)).ravel()
                residual = predicted_effect - pca_roundtrip
                pred_X[mask] = ctrl_means_val[ct] + sampled_effects + residual
            else:
                pred_X[mask] = ctrl_means_val[ct] + sampled_effects

    pred.X = pred_X
    return pred


# ===============================================================================
# METRICS COLLECTION
# ===============================================================================

def collect_all_metrics(ev, model_names):
    """Collect all 10 metrics into a summary DataFrame.

    COPIED FROM: run_la_reproduced.py lines 256-284
    """
    configs = [
        ('average', 'rmse', False, 'rmse_avg'),
        ('average', 'rmse', True, 'rmse_rank'),
        ('logfc', 'cosine', False, 'cos_logfc'),
        ('logfc', 'cosine', True, 'cos_rank'),
        ('pca_average', 'cosine', False, 'pca_cos'),
        ('pca_average', 'cosine', True, 'pca_cos_rank'),
        ('scores', 'r2_score', False, 'r2_score'),
        ('scores', 'top_k_recall', False, 'top_k_recall'),
        ('pca', 'mmd', False, 'mmd_pca'),
        ('pca', 'mmd', True, 'mmd_rank'),
    ]
    rows = {}
    for model in model_names:
        row = {}
        for aggr, metric, is_rank, col_name in configs:
            try:
                if is_rank:
                    df = ev.rank_evals[aggr][metric]
                    row[col_name] = df[df['model'] == model]['rank'].mean()
                else:
                    df = ev.evals[aggr][metric]
                    row[col_name] = df[df['model'] == model]['metric'].mean()
            except (KeyError, Exception):
                row[col_name] = float('nan')
        rows[model] = row
    return pd.DataFrame(rows).T


def run_extended_eval(ev):
    """Run pca_average/cosine, pca/mmd evaluation pipelines."""
    # pca_average / cosine + rank
    logger.info("  Extended eval: pca_average / cosine + rank...")
    try:
        ev.aggregate(aggr_method='pca_average')
        ev.evaluate(aggr_method='pca_average', metric='cosine')
        ev.evaluate_pairwise(aggr_method='pca_average', metric='cosine')
        ev.evaluate_rank(aggr_method='pca_average', metric='cosine')
        logger.info("    Done")
    except Exception as e:
        logger.info(f"    FAILED: {e}")

    # scores / r2_score + top_k_recall (disabled - threading deadlock)
    logger.info("  Extended eval: SKIPPED (scores / r2_score + top_k_recall - causes deadlock)")

    # pca / mmd + rank
    logger.info("  Extended eval: pca / mmd + rank...")
    try:
        ev.aggregate(aggr_method='pca')
        ev.evaluate(aggr_method='pca', metric='mmd')
        ev.evaluate_pairwise(aggr_method='pca', metric='mmd')
        ev.evaluate_rank(aggr_method='pca', metric='mmd')
        logger.info("    Done")
    except Exception as e:
        logger.info(f"    FAILED: {e}")


# ===============================================================================
# MAIN PIPELINE
# ===============================================================================

def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Log hyperparameters
    logger.info("=" * 80)
    logger.info("LA+FMM: Latent Additive + Finite Mixture Model")
    logger.info("=" * 80)
    logger.info(f"ENCODER_WIDTH={ENCODER_WIDTH}, LATENT_DIM={LATENT_DIM}, N_LAYERS={N_LAYERS}")
    logger.info(f"DROPOUT={DROPOUT}, LR={LR}, WD={WD}")
    logger.info(f"BATCH_SIZE={BATCH_SIZE}, MAX_EPOCHS={MAX_EPOCHS}")
    logger.info(f"INJECT_COVARIATES={INJECT_COVARIATES}, USE_CTRL_MEAN={USE_CTRL_MEAN}, PREDICT_PERCELL={PREDICT_PERCELL}")
    logger.info(f"LAMBDA_CORAL={LAMBDA_CORAL}, LAMBDA_CONTRASTIVE={LAMBDA_CONTRASTIVE}")
    logger.info(f"N_PCA={N_PCA}, K_VALUES={K_VALUES}, USE_RESIDUAL={USE_RESIDUAL}")
    logger.info("=" * 80)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"Device: {device} (Apple Metal)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        logger.info(f"Device: {device}")
        logger.info(f"GPU: {gpu_name} (count: {gpu_count})")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        logger.info(f"Device: {device} (no GPU detected)")
    logger.info("=" * 80)

    # ---- Step 1: Load data ----
    logger.info("\nStep 1: Loading data...")
    evaluator = Evaluator(
        task=TASK, local_data_cache=DATA_CACHE, split_value_to_evaluate='test')
    val_ref = evaluator.ref_adata
    split_dict = evaluator.get_split()

    full_adata = Evaluator.get_task_data(TASK, local_data_cache=DATA_CACHE)
    train_adata = full_adata[split_dict['train']].to_memory()
    val_adata = full_adata[split_dict['val']].to_memory()

    pert_col = evaluator.task_config.perturbation_key
    ctrl = evaluator.task_config.perturbation_control_value
    ct_col = evaluator.task_config.covariate_keys[0]

    logger.info(f"  Train: {train_adata.shape}, Val: {val_adata.shape}, Val ref: {val_ref.shape}")
    logger.info(f"  Pert col: {pert_col}, Ctrl: {ctrl}, CT col: {ct_col}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 2: Vocabularies ----
    logger.info("\nStep 2: Building vocabularies...")
    n_genes = train_adata.shape[1]
    all_perts = sorted([p for p in full_adata.obs[pert_col].unique() if p != ctrl])
    all_cts = sorted(full_adata.obs[ct_col].unique())
    pert2idx = {p: i for i, p in enumerate(all_perts)}
    ct2idx = {c: i for i, c in enumerate(all_cts)}
    n_perts = len(all_perts)
    n_cts = len(all_cts)
    logger.info(f"  {n_perts} perturbations, {n_cts} cell types, {n_genes} genes")

    # ---- Step 3: Train or load LA model ----
    logger.info("\nStep 3: LA model training/loading...")
    la_model, la_checkpoint = load_or_train_la_model(
        train_adata, val_adata, pert_col, ct_col, ctrl,
        pert2idx, ct2idx, all_perts, all_cts, n_genes, n_perts, n_cts, device)
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 4: Compute cell effects + PCA ----
    logger.info("\nStep 4: Computing per-cell effects and PCA...")
    cell_effects, mean_effects, ctrl_means_train = compute_cell_effects(
        train_adata, pert_col, ct_col, ctrl)
    total_cells = sum(v.shape[0] for v in cell_effects.values())
    logger.info(f"  {len(cell_effects)} (pert, ct) groups, {total_cells} cells")

    # Compute val control means
    ctrl_means_val = {}
    for ct in val_ref.obs[ct_col].unique():
        mask = (val_ref.obs[pert_col] == ctrl) & (val_ref.obs[ct_col] == ct)
        if mask.sum() > 0:
            ctrl_means_val[ct] = to_dense(val_ref[mask].X).mean(axis=0)

    # PCA
    logger.info(f"  PCA on all per-cell training effects (n_components={N_PCA})...")
    all_effects = np.concatenate(list(cell_effects.values()), axis=0)
    logger.info(f"  Effect matrix shape: {all_effects.shape}")
    pca = PCA(n_components=N_PCA)
    pca.fit(all_effects)
    logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Project all cell effects into PCA space
    projected_effects = {}
    for k, v in cell_effects.items():
        projected_effects[k] = pca.transform(v)
    del all_effects
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 5: Compute perturbation response similarity ----
    logger.info("\nStep 5: Computing perturbation-response similarities...")
    pert_sims = compute_pert_response_similarity(mean_effects)
    logger.info(f"  {len(pert_sims)} CT pairs")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 6: Prepare test control data for PREDICT_PERCELL ----
    logger.info("\nStep 6: Preparing test control data for PREDICT_PERCELL...")
    val_ctrl_mask = val_ref.obs[pert_col] == ctrl
    val_ctrl_expr = to_dense(val_ref[val_ctrl_mask].X)
    val_ctrl_ct = val_ref[val_ctrl_mask].obs[ct_col].values

    # Organize by cell type for PREDICT_PERCELL
    val_ctrl_data = {}
    for ct in all_cts:
        ct_cells = val_ctrl_expr[val_ctrl_ct == ct]
        val_ctrl_data[ct] = ct_cells  # (n_ctrl_for_ct, n_genes)
        logger.info(f"  {ct}: {len(ct_cells)} control cells")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 7: Build LA+FMM predictions ----
    logger.info("\nStep 7: Building LA+FMM predictions...")
    la_fmm_args = dict(
        val_ref=val_ref, cell_effects=cell_effects,
        ctrl_means_val=ctrl_means_val,
        pca=pca, projected_effects=projected_effects, pert_sims=pert_sims,
        la_model=la_model, la_checkpoint=la_checkpoint, val_ctrl_data=val_ctrl_data,
        pert_col=pert_col, ct_col=ct_col, ctrl=ctrl, device=device)

    model_preds = {}
    for K in K_VALUES:
        logger.info(f"  Building la_fmm_k{K}...")
        t1 = time.time()
        pred = build_la_fmm_prediction(**la_fmm_args, K=K)
        logger.info(f"    Done ({time.time()-t1:.1f}s)")
        model_preds[f'la_fmm_k{K}'] = pred

    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 8: Evaluate ----
    logger.info("\nStep 8: Running core 4 metrics...")
    core_metrics = evaluator.evaluate(
        model_predictions=model_preds, return_metrics_dataframe=True)
    logger.info(f"\nCore 4 metrics:\n{core_metrics}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # Extended evaluation
    logger.info("\nStep 9: Running extended metrics...")
    run_extended_eval(evaluator.ev)
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 10: Collect & print results ----
    model_names = list(model_preds.keys())
    all_metrics = collect_all_metrics(evaluator.ev, model_names)

    logger.info("\n" + "=" * 80)
    logger.info("ALL PERTURBENCH METRICS — LA+FMM")
    logger.info("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)
    logger.info(f"\n{all_metrics}")

    results_path = os.path.join(run_dir, 'results.csv')
    all_metrics.to_csv(results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Published baselines for comparison
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO PUBLISHED BASELINES")
    logger.info("=" * 80)
    logger.info("LA vanilla (rmse_avg=0.019, cos_logfc=0.52, mmd_pca=1.72)")
    logger.info("LFMM-K2    (rmse_avg=0.025, cos_logfc=0.21, mmd_pca=0.33)")
    logger.info("=" * 80)
    logger.info("Expected LA+FMM-K2: rmse_avg ~0.019-0.021, cos_logfc ~0.50-0.53, mmd_pca ~0.33-0.40")
    logger.info("Goal: Best of both worlds (LA mean quality + LFMM distributional quality)")

    total_time = time.time() - t0
    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
