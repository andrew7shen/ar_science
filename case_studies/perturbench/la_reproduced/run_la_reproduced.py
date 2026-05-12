"""
Reproduced Latent Additive (LA) model for Perturbation Effect Prediction on Srivatsan20.
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

warnings.filterwarnings('ignore')

from perturbench.analysis.benchmarks.evaluator import Evaluator
from perturbench.modelcore.nn.mlp import MLP

import scanpy as sc
sc.settings.verbosity = 2

# ---- Paths & Task ----
DATA_CACHE = 'eval/perturbench_eval/perturbench_files/data'
TASK = 'srivatsan20-transfer'
OUT_DIR = 'case_studies/perturbench/la_reproduced'

# ---- LA hyperparameters (from best params config) ----
ENCODER_WIDTH = 5376
LATENT_DIM = 192
N_LAYERS = 2
DROPOUT = 0.3
LR = 3.4e-5
WD = 6.1e-11
BATCH_SIZE = 4000
MAX_EPOCHS = 500

# ---- LA variant toggles ----
INJECT_COVARIATES = True   # True = concat ct_onehot to encoder/decoder (published LA)
USE_CTRL_MEAN = False      # True = per-CT mean ctrl expr; False = per-cell sampled (published LA)
PREDICT_PERCELL = True     # True = pass individual ctrl cells at inference (published); False = CT-mean

# ---- Training ----
PATIENCE_LR = 15
PATIENCE_EARLY = 50
SEED = 245

# ---- Pipeline control ----
REUSE_CHECKPOINT = False
CHECKPOINT_DIR = os.path.join(OUT_DIR, 'results', 'checkpoints')
CHECKPOINT_NAME = None  # Set to load a specific checkpoint, e.g. 'la_model_seed245.pt'
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

# --- Logging setup ---
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(OUT_DIR, 'results', f'la_{RUN_TIMESTAMP}_seed{SEED}')
os.makedirs(run_dir, exist_ok=True)

logger = logging.getLogger('la_reproduced')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(os.path.join(run_dir, 'run.log'), mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)


def to_dense(X):
    if issparse(X):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


# ---- Dataset ----

class LADataset(Dataset):
    """Per-cell dataset for LA training. Returns control expression for the cell's
    cell type, perturbation/cell-type one-hots, indices, and target expression."""

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


# ---- Model ----

class LatentAdditiveModel(nn.Module):
    def __init__(self, n_genes, n_perts, n_cts, inject_covariates=INJECT_COVARIATES):
        super().__init__()
        self.inject_covariates = inject_covariates
        enc_in = n_genes + n_cts if inject_covariates else n_genes
        dec_in = LATENT_DIM + n_cts if inject_covariates else LATENT_DIM
        self.gene_encoder = MLP(enc_in, ENCODER_WIDTH, LATENT_DIM, N_LAYERS, DROPOUT)
        self.pert_encoder = MLP(
            n_perts, ENCODER_WIDTH, LATENT_DIM, N_LAYERS, DROPOUT)
        self.decoder = MLP(dec_in, ENCODER_WIDTH, n_genes, N_LAYERS, DROPOUT)

    def forward(self, ctrl_expr, pert_onehot, ct_onehot):
        enc_input = torch.cat([ctrl_expr, ct_onehot], dim=1) if self.inject_covariates else ctrl_expr
        latent_ctrl = self.gene_encoder(enc_input)
        latent_pert = self.pert_encoder(pert_onehot)
        latent_perturbed = latent_ctrl + latent_pert
        dec_input = torch.cat([latent_perturbed, ct_onehot], dim=1) if self.inject_covariates else latent_perturbed
        return F.softplus(self.decoder(dec_input))


# ---- Metrics ----

def collect_all_metrics(ev, model_names):
    """Collect all 10 metrics into a summary DataFrame."""
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


# ---- Main ----

def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Log hyperparameters
    logger.info("=" * 80)
    logger.info("Reproduced Latent Additive (LA) Model")
    logger.info("=" * 80)
    logger.info(f"ENCODER_WIDTH={ENCODER_WIDTH}, LATENT_DIM={LATENT_DIM}, N_LAYERS={N_LAYERS}")
    logger.info(f"DROPOUT={DROPOUT}, LR={LR}, WD={WD}")
    logger.info(f"BATCH_SIZE={BATCH_SIZE}, MAX_EPOCHS={MAX_EPOCHS}")
    logger.info(f"INJECT_COVARIATES={INJECT_COVARIATES}, USE_CTRL_MEAN={USE_CTRL_MEAN}, PREDICT_PERCELL={PREDICT_PERCELL}")
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
    logger.info("Step 1: Loading data...")
    evaluator = Evaluator(
        task=TASK, local_data_cache=DATA_CACHE, split_value_to_evaluate='test')
    val_ref = evaluator.ref_adata
    split_dict = evaluator.get_split()

    full_adata = Evaluator.get_task_data(TASK, local_data_cache=DATA_CACHE)
    train_adata = full_adata[split_dict['train']].to_memory()

    pert_col = evaluator.task_config.perturbation_key
    ctrl = evaluator.task_config.perturbation_control_value
    ct_col = evaluator.task_config.covariate_keys[0]

    logger.info(f"  Train: {train_adata.shape}, Val ref: {val_ref.shape}")
    logger.info(f"  Pert col: {pert_col}, Ctrl: {ctrl}, CT col: {ct_col}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 2: Prepare training & val data (using PerturBench splits) ----
    logger.info("\nStep 2: Preparing data...")
    n_genes = train_adata.shape[1]
    val_adata = full_adata[split_dict['val']].to_memory()

    # Vocabularies from full dataset for consistency
    all_perts = sorted([p for p in full_adata.obs[pert_col].unique() if p != ctrl])
    all_cts = sorted(full_adata.obs[ct_col].unique())
    pert2idx = {p: i for i, p in enumerate(all_perts)}
    ct2idx = {c: i for i, c in enumerate(all_cts)}
    n_perts = len(all_perts)
    n_cts = len(all_cts)
    logger.info(f"  {n_perts} perturbations, {n_cts} cell types, {n_genes} genes")

    def build_dataset(adata, split_name):
        """Build LADataset from an adata split (train or val)."""
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
        poh = np.zeros((len(expr), n_perts), dtype=np.float32)
        poh[np.arange(len(expr)), pert_idx] = 1.0
        coh = np.zeros((len(expr), n_cts), dtype=np.float32)
        coh[np.arange(len(expr)), ct_idx] = 1.0
        logger.info(f"  {split_name}: {len(expr)} perturbed cells")
        return LADataset(expr, ct_idx, pert_idx, poh, coh, cmeans,
                         ctrl_expr_all, ctrl_indices_by_ct), cmeans

    train_ds, ctrl_means_train = build_dataset(train_adata, "train")
    val_ds, ctrl_means_val = build_dataset(val_adata, "val")

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 4: Build model & train or load checkpoint ----
    logger.info("\nStep 4: Building model...")
    model = LatentAdditiveModel(n_genes, n_perts, n_cts).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME) if CHECKPOINT_NAME else None

    if REUSE_CHECKPOINT and checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"  Loaded: val_mse={checkpoint['val_mse']:.4f}, seed={checkpoint.get('seed', '?')}")

        assert checkpoint['n_genes'] == n_genes
        assert checkpoint['n_perts'] == n_perts
        assert checkpoint['n_cts'] == n_cts

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info(f"  Time: {time.time()-t0:.1f}s")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=PATIENCE_LR, factor=0.2)

        # ---- Step 5: Training loop ----
        logger.info(f"\nStep 5: Training (max {MAX_EPOCHS} epochs, "
            f"early stop patience={PATIENCE_EARLY})...")
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

                pred = model(ctrl_expr, pert_oh, ct_oh)
                loss = F.mse_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_mse_losses.append(loss.item())

            train_mse = np.mean(train_mse_losses)

            # -- Validate --
            model.eval()
            val_mse_losses = []
            with torch.no_grad():
                for batch in val_dl:
                    ctrl_expr, pert_oh, ct_oh, ct_idx, pert_idx, target = \
                        [b.to(device) for b in batch]
                    pred = model(ctrl_expr, pert_oh, ct_oh)
                    val_mse_losses.append(F.mse_loss(pred, target).item())

            val_mse = np.mean(val_mse_losses)

            scheduler.step(val_mse)

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
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        logger.info(f"  Time: {time.time()-t0:.1f}s")

        # Save checkpoint
        checkpoint = {
            'model_state_dict': best_state,
            'n_genes': n_genes,
            'n_perts': n_perts,
            'n_cts': n_cts,
            'pert2idx': pert2idx,
            'ct2idx': ct2idx,
            'val_mse': best_val_loss,
            'seed': SEED,
            'hyperparams': {
                'encoder_width': ENCODER_WIDTH,
                'latent_dim': LATENT_DIM,
                'n_layers': N_LAYERS,
                'dropout': DROPOUT,
                'inject_covariates': INJECT_COVARIATES,
            }
        }
        save_name = f'la_model_seed{SEED}.pt'
        save_path = os.path.join(CHECKPOINT_DIR, save_name)
        torch.save(checkpoint, save_path)
        logger.info(f"  Saved checkpoint to {save_path}")

    # ---- Step 6: Generate predictions for PerturBench test set ----
    logger.info(f"\nStep 6: Generating predictions for PerturBench test set "
                f"(PREDICT_PERCELL={PREDICT_PERCELL})...")

    MAX_CTRL_CELLS = 1000  # match PerturBench max_control_cells_per_covariate
    pred_adata = val_ref.copy()
    pred_X = to_dense(pred_adata.X).copy()
    rng = np.random.RandomState(SEED)

    if PREDICT_PERCELL:
        # PerturBench approach: pass individual control cells through model,
        # keep per-cell predictions (one per sampled control cell).
        # Control cells come from the test split reference.
        ref_ctrl_mask = val_ref.obs[pert_col] == ctrl
        ref_ctrl_expr = to_dense(val_ref[ref_ctrl_mask].X)
        ref_ctrl_ct = val_ref[ref_ctrl_mask].obs[ct_col].values

        with torch.no_grad():
            for ct in val_ref.obs[ct_col].unique():
                ct_idx_val = ct2idx.get(ct)
                if ct_idx_val is None:
                    continue
                ct_mask = val_ref.obs[ct_col] == ct
                ct_oh = np.zeros(n_cts, dtype=np.float32)
                ct_oh[ct_idx_val] = 1.0

                # Sample up to MAX_CTRL_CELLS control cells for this CT
                ct_ctrl_cells = ref_ctrl_expr[ref_ctrl_ct == ct]
                n_ctrl_available = len(ct_ctrl_cells)
                if n_ctrl_available == 0:
                    continue
                if n_ctrl_available > MAX_CTRL_CELLS:
                    sample_idxs = rng.choice(n_ctrl_available, MAX_CTRL_CELLS, replace=False)
                    ct_ctrl_cells = ct_ctrl_cells[sample_idxs]
                n_ctrl = len(ct_ctrl_cells)
                logger.info(f"  CT={ct}: {n_ctrl} control cells (of {n_ctrl_available})")

                ct_oh_batch = np.tile(ct_oh, (n_ctrl, 1))
                ctrl_t = torch.tensor(ct_ctrl_cells, dtype=torch.float32).to(device)
                ct_oh_t = torch.tensor(ct_oh_batch, dtype=torch.float32).to(device)

                for pert in val_ref.obs.loc[ct_mask, pert_col].unique():
                    if pert == ctrl or pert not in pert2idx:
                        continue
                    pert_oh = np.zeros(n_perts, dtype=np.float32)
                    pert_oh[pert2idx[pert]] = 1.0
                    pert_oh_batch = np.tile(pert_oh, (n_ctrl, 1))
                    pert_oh_t = torch.tensor(pert_oh_batch, dtype=torch.float32).to(device)

                    # Pass control cells through model, keep per-cell predictions
                    pred_all = model(ctrl_t, pert_oh_t, ct_oh_t)
                    pred_np = pred_all.cpu().numpy()

                    cell_mask = (ct_mask & (val_ref.obs[pert_col] == pert)).values
                    n_cells = cell_mask.sum()
                    # Sample from per-control-cell predictions to fill observed cells
                    sample_idx = rng.choice(n_ctrl, size=n_cells, replace=True)
                    pred_X[cell_mask] = pred_np[sample_idx]
    else:
        # CT-mean approach: f(E[x]). Use per-CT mean control expression.
        with torch.no_grad():
            for ct in val_ref.obs[ct_col].unique():
                ct_idx_val = ct2idx.get(ct)
                if ct_idx_val is None:
                    continue
                ct_mask = val_ref.obs[ct_col] == ct
                ct_oh = np.zeros(n_cts, dtype=np.float32)
                ct_oh[ct_idx_val] = 1.0
                ctrl_t = torch.tensor(
                    ctrl_means_val[ct_idx_val], dtype=torch.float32
                ).unsqueeze(0).to(device)
                ct_oh_t = torch.tensor(ct_oh).unsqueeze(0).to(device)

                for pert in val_ref.obs.loc[ct_mask, pert_col].unique():
                    if pert == ctrl or pert not in pert2idx:
                        continue
                    pert_oh = np.zeros(n_perts, dtype=np.float32)
                    pert_oh[pert2idx[pert]] = 1.0
                    pert_oh_t = torch.tensor(pert_oh).unsqueeze(0).to(device)

                    pred_expr = model(ctrl_t, pert_oh_t, ct_oh_t)
                    pred_np = pred_expr.cpu().numpy().flatten()

                    cell_mask = (ct_mask & (val_ref.obs[pert_col] == pert)).values
                    n_cells = cell_mask.sum()
                    noise = rng.normal(0, 1e-6, size=(n_cells, n_genes))
                    pred_X[cell_mask] = pred_np + noise

    pred_adata.X = pred_X
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 7: Evaluate ----
    logger.info("\nStep 7: Running core 4 metrics...")
    model_name = 'la_reproduced'
    model_preds = {model_name: pred_adata}

    core_metrics = evaluator.evaluate(
        model_predictions=model_preds, return_metrics_dataframe=True)
    logger.info("\nCore 4 metrics:")
    logger.info(f"Core 4 metrics:\n{core_metrics}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # Extended metrics
    ev = evaluator.ev
    model_names = [model_name]

    logger.info("\nStep 7a: pca_average / cosine + rank...")
    try:
        ev.aggregate(aggr_method='pca_average')
        ev.evaluate(aggr_method='pca_average', metric='cosine')
        ev.evaluate_pairwise(aggr_method='pca_average', metric='cosine')
        ev.evaluate_rank(aggr_method='pca_average', metric='cosine')
        logger.info("  Done")
    except Exception as e:
        logger.info(f"  FAILED: {e}")

    logger.info("\nStep 7b: pca / mmd + rank...")
    try:
        ev.aggregate(aggr_method='pca')
        ev.evaluate(aggr_method='pca', metric='mmd')
        ev.evaluate_pairwise(aggr_method='pca', metric='mmd')
        ev.evaluate_rank(aggr_method='pca', metric='mmd')
        logger.info("  Done")
    except Exception as e:
        logger.info(f"  FAILED: {e}")

    # ---- Step 8: Results ----
    all_metrics = collect_all_metrics(ev, model_names)

    logger.info("\n" + "=" * 80)
    logger.info("ALL PERTURBENCH METRICS")
    logger.info("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)
    logger.info(f"\n{all_metrics}")

    results_path = os.path.join(run_dir, 'results.csv')
    all_metrics.to_csv(results_path)
    logger.info(f"Results saved to {results_path}")

    total_time = time.time() - t0
    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
