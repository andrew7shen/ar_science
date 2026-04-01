"""
FMM (Finite Mixture Model) Baseline for PerturBench Srivatsan20.

Tests whether GMM-based perturbation effect sampling captures single-cell
response heterogeneity better than a single mean effect. Twelve baselines:
  - Mean (K=1): single average effect
  - Sample: random individual-cell effect
  - GMM-K2/K3/K5: GMM-sampled effects with 2/3/5 components (pooled proportions)
  - FMM-K2/K3/K5: GMM components with cell-type-specific mixing proportions
  - Linear (K=1): per-gene OLS transfer, no GMM
  - LFMM-K2/K3/K5: linear mean + GMM distributional structure
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import logging
import warnings
import time
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

from perturbench.analysis.benchmarks.evaluator import Evaluator
import scanpy as sc
sc.settings.verbosity = 2

DATA_CACHE = 'eval/perturbench_eval/perturbench_files/data'
TASK = 'srivatsan20-transfer'
OUT_DIR = 'eval/perturbench_eval/fmm_baseline'
N_PCA = 50
TOY_MODE = True  # Set to True to test with only K=2

# --- Logging setup ---
log_dir = os.path.join(OUT_DIR, 'results')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('fmm')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(os.path.join(log_dir, 'run.log'), mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)


def to_dense(X):
    if issparse(X):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def compute_cell_effects(train_adata, pert_col, ct_col, ctrl):
    """Compute per-cell effects and control means per cell type."""
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


def compute_ct_deltas(mean_effects, pca):
    """Mean effect difference (target - source) per CT pair, in PCA space.

    For each (ct_target, ct_source) pair, average [effect(p, ct_target) - effect(p, ct_source)]
    across shared perturbations, then project to PCA space.
    """
    ct_perts = {}
    for (pert, ct) in mean_effects:
        ct_perts.setdefault(ct, set()).add(pert)

    cts = list(ct_perts.keys())
    deltas = {}
    for ct_target in cts:
        for ct_source in cts:
            if ct_target == ct_source:
                continue
            shared = ct_perts[ct_target] & ct_perts[ct_source]
            if not shared:
                deltas[(ct_target, ct_source)] = np.zeros((1, pca.n_components_))
                continue
            diffs = [mean_effects[(p, ct_target)] - mean_effects[(p, ct_source)] for p in shared]
            avg_diff = np.mean(diffs, axis=0)
            deltas[(ct_target, ct_source)] = pca.transform(avg_diff.reshape(1, -1))
    return deltas


def learn_linear_transfer(mean_effects):
    """Per-gene OLS: Y_target = W * X_source + b for each (target, source) CT pair."""
    ct_perts = {}
    for (pert, ct) in mean_effects:
        ct_perts.setdefault(ct, set()).add(pert)
    cts = list(ct_perts.keys())
    linear_models = {}
    for ct_target in cts:
        for ct_source in cts:
            if ct_target == ct_source:
                continue
            shared = sorted(ct_perts[ct_target] & ct_perts[ct_source])
            if len(shared) < 3:
                continue
            X = np.stack([mean_effects[(p, ct_source)] for p in shared])
            Y = np.stack([mean_effects[(p, ct_target)] for p in shared])
            x_mean, y_mean = X.mean(0), Y.mean(0)
            x_centered = X - x_mean
            var_x = (x_centered ** 2).mean(0) + 1e-10
            W = (x_centered * (Y - y_mean)).mean(0) / var_x
            b = y_mean - W * x_mean
            linear_models[(ct_target, ct_source)] = (W, b)
    return linear_models


def _add_noise(pred_X, mask, predicted_expr, rng):
    n_cells = mask.sum()
    noise = rng.normal(0, 1e-6, size=(n_cells, predicted_expr.shape[-1] if predicted_expr.ndim > 1 else len(predicted_expr)))
    if predicted_expr.ndim == 1:
        pred_X[mask] = predicted_expr + noise
    else:
        pred_X[mask] = predicted_expr + noise


def build_mean_prediction(val_ref, mean_effects, ctrl_means_val, pert_col, ct_col, ctrl):
    """Mean baseline: transfer average effect from other cell types."""
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            other_effects = [mean_effects[(pert, other_ct)]
                             for other_ct in ctrl_means_val
                             if other_ct != ct and (pert, other_ct) in mean_effects]
            if not other_effects:
                continue
            predicted_expr = ctrl_means_val[ct] + np.mean(other_effects, axis=0)
            mask = (ct_mask & (obs[pert_col] == pert)).values
            _add_noise(pred_X, mask, predicted_expr, rng)

    pred.X = pred_X
    return pred


def build_linear_prediction(val_ref, mean_effects, ctrl_means_val, pert_sims,
                             linear_models, pert_col, ct_col, ctrl):
    """Linear baseline: per-gene OLS transfer weighted by pert-response similarity."""
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            source_cts = [sct for sct in ctrl_means_val
                          if sct != ct and (pert, sct) in mean_effects
                          and (ct, sct) in linear_models]
            if not source_cts:
                continue
            ct_weights = np.array([pert_sims.get((ct, sct), 0.5) for sct in source_cts])
            ct_weights /= ct_weights.sum()
            predicted_effect = np.zeros_like(ctrl_means_val[ct])
            for i, sct in enumerate(source_cts):
                W, b = linear_models[(ct, sct)]
                pred_sct = W * mean_effects[(pert, sct)] + b
                predicted_effect += ct_weights[i] * pred_sct
            predicted_expr = ctrl_means_val[ct] + predicted_effect
            mask = (ct_mask & (obs[pert_col] == pert)).values
            _add_noise(pred_X, mask, predicted_expr, rng)

    pred.X = pred_X
    return pred


def build_sample_prediction(val_ref, cell_effects, ctrl_means_val, pert_col, ct_col, ctrl):
    """Sample baseline: randomly sample one training cell effect per val cell."""
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            pooled = [cell_effects[(pert, other_ct)]
                      for other_ct in ctrl_means_val
                      if other_ct != ct and (pert, other_ct) in cell_effects]
            if not pooled:
                continue
            pooled_effects = np.concatenate(pooled, axis=0)
            mask = (ct_mask & (obs[pert_col] == pert)).values
            n_cells = mask.sum()
            idxs = rng.choice(len(pooled_effects), size=n_cells, replace=True)
            sampled = pooled_effects[idxs]
            pred_X[mask] = ctrl_means_val[ct] + sampled

    pred.X = pred_X
    return pred


def build_gmm_prediction(val_ref, cell_effects, ctrl_means_val, pca, projected_effects,
                          K, pert_col, ct_col, ctrl):
    """GMM baseline: fit GMM on PCA-projected pooled effects, sample per val cell."""
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)
    min_cells_for_gmm = 20

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            # Pool effects from other cell types
            pooled_pca = [projected_effects[(pert, other_ct)]
                          for other_ct in ctrl_means_val
                          if other_ct != ct and (pert, other_ct) in projected_effects]
            pooled_raw = [cell_effects[(pert, other_ct)]
                          for other_ct in ctrl_means_val
                          if other_ct != ct and (pert, other_ct) in cell_effects]
            if not pooled_pca:
                continue

            pooled_pca_arr = np.concatenate(pooled_pca, axis=0)
            pooled_raw_arr = np.concatenate(pooled_raw, axis=0)
            n_pooled = len(pooled_pca_arr)
            mask = (ct_mask & (obs[pert_col] == pert)).values
            n_cells = mask.sum()

            # Fallback to mean if too few cells
            if n_pooled < min_cells_for_gmm:
                mean_effect = pooled_raw_arr.mean(axis=0)
                pred_X[mask] = ctrl_means_val[ct] + mean_effect + rng.normal(0, 1e-6, size=(n_cells, pred_X.shape[1]))
                logger.debug(f"  ({pert}, {ct}): {n_pooled} cells -> fallback to mean")
                continue

            actual_k = min(K, n_pooled // 5)
            actual_k = max(actual_k, 1)

            gmm = GaussianMixture(n_components=actual_k, covariance_type='diag',
                                  random_state=42, max_iter=200)
            gmm.fit(pooled_pca_arr)
            sampled_pca, _ = gmm.sample(n_cells)
            sampled_effects = pca.inverse_transform(sampled_pca.astype(np.float32))
            pred_X[mask] = ctrl_means_val[ct] + sampled_effects

    pred.X = pred_X
    return pred


def build_fmm_prediction(val_ref, cell_effects, ctrl_means_val,
                          pca, projected_effects, pert_sims, ct_deltas,
                          K, pert_col, ct_col, ctrl):
    """FMM baseline: CT-specific mixing proportions + component mean shift."""
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)
    min_cells_for_gmm = 20

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            # Pool effects from other cell types (same as GMM)
            source_cts = [other_ct for other_ct in ctrl_means_val
                          if other_ct != ct and (pert, other_ct) in projected_effects]
            if not source_cts:
                continue

            pooled_pca_arr = np.concatenate([projected_effects[(pert, sct)] for sct in source_cts], axis=0)
            pooled_raw_arr = np.concatenate([cell_effects[(pert, sct)] for sct in source_cts], axis=0)
            n_pooled = len(pooled_pca_arr)
            mask = (ct_mask & (obs[pert_col] == pert)).values
            n_cells = mask.sum()

            # Fallback to mean if too few cells
            if n_pooled < min_cells_for_gmm:
                mean_effect = pooled_raw_arr.mean(axis=0)
                pred_X[mask] = ctrl_means_val[ct] + mean_effect + rng.normal(0, 1e-6, size=(n_cells, pred_X.shape[1]))
                continue

            actual_k = min(K, n_pooled // 5)
            actual_k = max(actual_k, 1)

            # Fit GMM on pooled effects (shared components)
            gmm = GaussianMixture(n_components=actual_k, covariance_type='diag',
                                  random_state=42, max_iter=200)
            gmm.fit(pooled_pca_arr)

            # Perturbation-response-based CT weights
            ct_weights = np.array([pert_sims.get((ct, sct), 0.5) for sct in source_cts])
            ct_weights /= ct_weights.sum()

            # Per-source-CT mixing proportions
            pi_per_ct = []
            for sct in source_cts:
                assignments = gmm.predict(projected_effects[(pert, sct)])
                pi = np.bincount(assignments, minlength=actual_k) / len(assignments)
                pi_per_ct.append(pi)

            # Weighted mixture of per-CT proportions
            target_pi = sum(w * pi for w, pi in zip(ct_weights, pi_per_ct))
            target_pi /= target_pi.sum()
            gmm.weights_ = target_pi

            # Shift component means by weighted delta
            weighted_delta = sum(ct_weights[i] * ct_deltas.get((ct, sct), np.zeros((1, pca.n_components_)))
                                 for i, sct in enumerate(source_cts))
            gmm.means_ = gmm.means_ + weighted_delta  # (K, N_PCA) + (1, N_PCA)

            # Sample with corrected means and proportions
            sampled_pca, _ = gmm.sample(n_cells)
            sampled_effects = pca.inverse_transform(sampled_pca.astype(np.float32))
            pred_X[mask] = ctrl_means_val[ct] + sampled_effects

            logger.debug(f"  FMM ({pert}, {ct}): K={actual_k}, "
                         f"source_cts={source_cts}, ct_weights={ct_weights.round(3)}, "
                         f"target_pi={target_pi.round(3)}")

    pred.X = pred_X
    return pred


def build_linear_fmm_prediction(val_ref, cell_effects, ctrl_means_val,
                                  pca, projected_effects, pert_sims,
                                  linear_models, mean_effects,
                                  K, pert_col, ct_col, ctrl):
    """Linear-FMM hybrid: linear mean + GMM distributional structure."""
    pred = val_ref.copy()
    pred_X = to_dense(pred.X).copy()
    obs = val_ref.obs
    rng = np.random.RandomState(42)
    min_cells_for_gmm = 20

    for ct in obs[ct_col].unique():
        if ct not in ctrl_means_val:
            continue
        ct_mask = obs[ct_col] == ct
        for pert in obs.loc[ct_mask, pert_col].unique():
            if pert == ctrl:
                continue
            source_cts = [sct for sct in ctrl_means_val
                          if sct != ct and (pert, sct) in projected_effects]
            if not source_cts:
                continue

            pooled_pca_arr = np.concatenate([projected_effects[(pert, sct)] for sct in source_cts], axis=0)
            n_pooled = len(pooled_pca_arr)
            mask = (ct_mask & (obs[pert_col] == pert)).values
            n_cells = mask.sum()

            # CT weights from pert-response similarity
            ct_weights = np.array([pert_sims.get((ct, sct), 0.5) for sct in source_cts])
            ct_weights /= ct_weights.sum()

            # Linear mean prediction (gene space)
            linear_cts = [sct for sct in source_cts if (ct, sct) in linear_models]
            if linear_cts:
                lin_weights = np.array([pert_sims.get((ct, sct), 0.5) for sct in linear_cts])
                lin_weights /= lin_weights.sum()
                predicted_effect = np.zeros_like(ctrl_means_val[ct])
                for i, sct in enumerate(linear_cts):
                    W, b = linear_models[(ct, sct)]
                    predicted_effect += lin_weights[i] * (W * mean_effects[(pert, sct)] + b)
            else:
                # Fallback: simple average of source effects
                predicted_effect = np.mean([mean_effects[(pert, sct)] for sct in source_cts], axis=0)

            # Fallback to linear-only if too few cells for GMM
            if n_pooled < min_cells_for_gmm:
                predicted_expr = ctrl_means_val[ct] + predicted_effect
                _add_noise(pred_X, mask, predicted_expr, rng)
                continue

            actual_k = min(K, n_pooled // 5)
            actual_k = max(actual_k, 1)

            # Fit GMM on pooled effects (same as FMM)
            gmm = GaussianMixture(n_components=actual_k, covariance_type='diag',
                                  random_state=42, max_iter=200)
            gmm.fit(pooled_pca_arr)

            # FMM proportions (same as FMM)
            pi_per_ct = []
            for sct in source_cts:
                assignments = gmm.predict(projected_effects[(pert, sct)])
                pi = np.bincount(assignments, minlength=actual_k) / len(assignments)
                pi_per_ct.append(pi)
            target_pi = sum(w * pi for w, pi in zip(ct_weights, pi_per_ct))
            target_pi /= target_pi.sum()
            gmm.weights_ = target_pi

            # Recenter GMM on linear prediction (replaces delta correction)
            mu_gmm = np.average(gmm.means_, axis=0, weights=target_pi)
            mu_linear = pca.transform(predicted_effect.reshape(1, -1)).ravel()
            gmm.means_ += (mu_linear - mu_gmm)

            # Sample from recentered GMM
            sampled_pca, _ = gmm.sample(n_cells)
            sampled_effects = pca.inverse_transform(sampled_pca.astype(np.float32))
            pred_X[mask] = ctrl_means_val[ct] + sampled_effects

    pred.X = pred_X
    return pred


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


def run_extended_eval(ev):
    """Run pca_average/cosine, scores/r2+top_k, pca/mmd evaluation pipelines."""
    # pca_average / cosine + rank
    logger.info("Extended eval: pca_average / cosine + rank...")
    try:
        t1 = time.time()
        ev.aggregate(aggr_method='pca_average')
        logger.info(f"  aggregated ({time.time()-t1:.1f}s)")
        ev.evaluate(aggr_method='pca_average', metric='cosine')
        logger.info(f"  evaluated ({time.time()-t1:.1f}s)")
        ev.evaluate_pairwise(aggr_method='pca_average', metric='cosine')
        logger.info(f"  pairwise ({time.time()-t1:.1f}s)")
        ev.evaluate_rank(aggr_method='pca_average', metric='cosine')
        logger.info(f"  rank ({time.time()-t1:.1f}s)")
    except Exception as e:
        logger.info(f"  FAILED: {e}")

    # scores / r2_score + top_k_recall (disabled - threading deadlock)
    logger.info("Extended eval: SKIPPED (scores / r2_score + top_k_recall - threading deadlock)")

    # pca / mmd + rank
    logger.info("Extended eval: pca / mmd + rank...")
    try:
        t1 = time.time()
        ev.aggregate(aggr_method='pca')
        logger.info(f"  aggregated ({time.time()-t1:.1f}s)")
        ev.evaluate(aggr_method='pca', metric='mmd')
        logger.info(f"  evaluated ({time.time()-t1:.1f}s)")
        ev.evaluate_pairwise(aggr_method='pca', metric='mmd')
        logger.info(f"  pairwise ({time.time()-t1:.1f}s)")
        ev.evaluate_rank(aggr_method='pca', metric='mmd')
        logger.info(f"  rank ({time.time()-t1:.1f}s)")
    except Exception as e:
        logger.info(f"  FAILED: {e}")


def main():
    t0 = time.time()

    # ---- GPU Check ----
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        logger.info("=" * 80)
        logger.info("GPU CHECK")
        logger.info("=" * 80)
        logger.info(f"GPU Available: {gpu_available}")
        logger.info(f"GPU Count: {gpu_count}")
        logger.info(f"GPU Name: {gpu_name}")
        logger.info("=" * 80)
    except ImportError:
        logger.info("PyTorch not available for GPU check")

    # ---- Step 1: Load data ----
    logger.info("Step 1: Loading data and creating splits...")
    evaluator = Evaluator(
        task=TASK,
        local_data_cache=DATA_CACHE,
        split_value_to_evaluate='val',
    )
    val_ref = evaluator.ref_adata
    split_dict = evaluator.get_split()

    full_adata = Evaluator.get_task_data(TASK, local_data_cache=DATA_CACHE)
    train_adata = full_adata[split_dict['train']].to_memory()

    pert_col = evaluator.task_config.perturbation_key
    ctrl = evaluator.task_config.perturbation_control_value
    ct_col = evaluator.task_config.covariate_keys[0]

    logger.info(f"  Train: {train_adata.shape}, Val ref: {val_ref.shape}")
    logger.info(f"  Pert col: {pert_col}, Ctrl: {ctrl}, Cell type col: {ct_col}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 2: Compute per-cell effects ----
    logger.info("Step 2: Computing per-cell perturbation effects...")
    cell_effects, mean_effects, ctrl_means_train = compute_cell_effects(
        train_adata, pert_col, ct_col, ctrl)
    total_cells = sum(v.shape[0] for v in cell_effects.values())
    logger.info(f"  {len(cell_effects)} (pert, ct) groups, {total_cells} total cells")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # Compute val control means
    ctrl_means_val = {}
    for ct in val_ref.obs[ct_col].unique():
        mask = (val_ref.obs[pert_col] == ctrl) & (val_ref.obs[ct_col] == ct)
        if mask.sum() > 0:
            ctrl_means_val[ct] = to_dense(val_ref[mask].X).mean(axis=0)

    # ---- Step 3: PCA on training effects ----
    logger.info(f"Step 3: PCA on all per-cell training effects (n_components={N_PCA})...")
    all_effects = np.concatenate(list(cell_effects.values()), axis=0)
    logger.info(f"  Effect matrix shape: {all_effects.shape}")
    pca = PCA(n_components=N_PCA)
    pca.fit(all_effects)
    logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # Project all cell effects into PCA space
    projected_effects = {}
    for k, v in cell_effects.items():
        projected_effects[k] = pca.transform(v)
    del all_effects

    # ---- Step 4: Build predictions ----
    if TOY_MODE:
        logger.info("Step 4: Building predictions for TOY MODE (only K=2 baselines)...")
    else:
        logger.info("Step 4: Building predictions for all 12 baselines...")

    # Mean (K=1)
    logger.info("  Building mean baseline...")
    pred_mean = build_mean_prediction(val_ref, mean_effects, ctrl_means_val,
                                      pert_col, ct_col, ctrl)
    logger.info(f"  Mean done. Time: {time.time()-t0:.1f}s")

    # Sample
    logger.info("  Building sample baseline...")
    pred_sample = build_sample_prediction(val_ref, cell_effects, ctrl_means_val,
                                          pert_col, ct_col, ctrl)
    logger.info(f"  Sample done. Time: {time.time()-t0:.1f}s")

    # GMM-K2
    logger.info("  Building GMM-K2 baseline...")
    pred_k2 = build_gmm_prediction(val_ref, cell_effects, ctrl_means_val, pca,
                                    projected_effects, 2, pert_col, ct_col, ctrl)
    logger.info(f"  GMM-K2 done. Time: {time.time()-t0:.1f}s")

    if not TOY_MODE:
        # GMM-K3
        logger.info("  Building GMM-K3 baseline...")
        pred_k3 = build_gmm_prediction(val_ref, cell_effects, ctrl_means_val, pca,
                                        projected_effects, 3, pert_col, ct_col, ctrl)
        logger.info(f"  GMM-K3 done. Time: {time.time()-t0:.1f}s")

        # GMM-K5
        logger.info("  Building GMM-K5 baseline...")
        pred_k5 = build_gmm_prediction(val_ref, cell_effects, ctrl_means_val, pca,
                                        projected_effects, 5, pert_col, ct_col, ctrl)
        logger.info(f"  GMM-K5 done. Time: {time.time()-t0:.1f}s")

    # FMM baselines (perturbation-response similarity + mean shift)
    logger.info("  Computing perturbation-response similarities...")
    pert_sims = compute_pert_response_similarity(mean_effects)
    logger.info(f"  Pert-response similarities: {len(pert_sims)} pairs")
    for (ct_a, ct_b), sim in sorted(pert_sims.items()):
        logger.debug(f"    pert_sim({ct_a}, {ct_b}) = {sim:.4f}")

    logger.info("  Computing CT deltas...")
    ct_deltas = compute_ct_deltas(mean_effects, pca)
    logger.info(f"  CT deltas: {len(ct_deltas)} pairs")

    fmm_args = dict(val_ref=val_ref, cell_effects=cell_effects,
                     ctrl_means_val=ctrl_means_val,
                     pca=pca, projected_effects=projected_effects,
                     pert_sims=pert_sims, ct_deltas=ct_deltas,
                     pert_col=pert_col, ct_col=ct_col, ctrl=ctrl)

    logger.info("  Building FMM-K2 baseline...")
    pred_fmm_k2 = build_fmm_prediction(**fmm_args, K=2)
    logger.info(f"  FMM-K2 done. Time: {time.time()-t0:.1f}s")

    if not TOY_MODE:
        logger.info("  Building FMM-K3 baseline...")
        pred_fmm_k3 = build_fmm_prediction(**fmm_args, K=3)
        logger.info(f"  FMM-K3 done. Time: {time.time()-t0:.1f}s")

        logger.info("  Building FMM-K5 baseline...")
        pred_fmm_k5 = build_fmm_prediction(**fmm_args, K=5)
        logger.info(f"  FMM-K5 done. Time: {time.time()-t0:.1f}s")

    # Linear + LFMM baselines
    logger.info("  Learning per-gene linear transfer models...")
    linear_models = learn_linear_transfer(mean_effects)
    logger.info(f"  Linear models: {len(linear_models)} CT pairs")

    logger.info("  Building linear_k1 baseline...")
    pred_linear = build_linear_prediction(val_ref, mean_effects, ctrl_means_val,
                                           pert_sims, linear_models,
                                           pert_col, ct_col, ctrl)
    logger.info(f"  linear_k1 done. Time: {time.time()-t0:.1f}s")

    lfmm_args = dict(val_ref=val_ref, cell_effects=cell_effects,
                      ctrl_means_val=ctrl_means_val,
                      pca=pca, projected_effects=projected_effects,
                      pert_sims=pert_sims, linear_models=linear_models,
                      mean_effects=mean_effects,
                      pert_col=pert_col, ct_col=ct_col, ctrl=ctrl)

    logger.info("  Building lfmm_k2 baseline...")
    pred_lfmm_k2 = build_linear_fmm_prediction(**lfmm_args, K=2)
    logger.info(f"  lfmm_k2 done. Time: {time.time()-t0:.1f}s")

    if not TOY_MODE:
        logger.info("  Building lfmm_k3 baseline...")
        pred_lfmm_k3 = build_linear_fmm_prediction(**lfmm_args, K=3)
        logger.info(f"  lfmm_k3 done. Time: {time.time()-t0:.1f}s")

        logger.info("  Building lfmm_k5 baseline...")
        pred_lfmm_k5 = build_linear_fmm_prediction(**lfmm_args, K=5)
        logger.info(f"  lfmm_k5 done. Time: {time.time()-t0:.1f}s")

    # ---- Step 5: Evaluate ----
    logger.info("Step 5: Running core 4 metrics via Evaluator...")
    if TOY_MODE:
        model_preds = {
            'mean_k1': pred_mean,
            'sample': pred_sample,
            'gmm_k2': pred_k2,
            'fmm_k2': pred_fmm_k2,
            'linear_k1': pred_linear,
            'lfmm_k2': pred_lfmm_k2,
        }
    else:
        model_preds = {
            'mean_k1': pred_mean,
            'sample': pred_sample,
            'gmm_k2': pred_k2,
            'gmm_k3': pred_k3,
            'gmm_k5': pred_k5,
            'fmm_k2': pred_fmm_k2,
            'fmm_k3': pred_fmm_k3,
            'fmm_k5': pred_fmm_k5,
            'linear_k1': pred_linear,
            'lfmm_k2': pred_lfmm_k2,
            'lfmm_k3': pred_lfmm_k3,
            'lfmm_k5': pred_lfmm_k5,
        }
    core_metrics = evaluator.evaluate(
        model_predictions=model_preds,
        return_metrics_dataframe=True,
    )
    logger.info(f"Core 4 metrics:\n{core_metrics}")
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # Extended evaluation
    logger.info("Step 6: Running extended metrics...")
    run_extended_eval(evaluator.ev)
    logger.info(f"  Time: {time.time()-t0:.1f}s")

    # ---- Step 6: Collect & print results ----
    model_names = list(model_preds.keys())
    if not TOY_MODE:
        all_metrics = collect_all_metrics(evaluator.ev, model_names)

    logger.info("\n" + "=" * 80)
    if TOY_MODE:
        logger.info("TOY MODE RESULTS — FMM BASELINES (K=2 ONLY)")
    else:
        logger.info("ALL PERTURBENCH METRICS — FMM BASELINES")
    logger.info("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)

    if TOY_MODE:
        logger.info(f"\nCore metrics:\n{core_metrics}")
        core_metrics.to_csv(os.path.join(log_dir, 'results_toy_mode.csv'))
        logger.info(f"Results saved to {log_dir}/results_toy_mode.csv")
    else:
        logger.info(f"\n{all_metrics}")
        all_metrics.to_csv(os.path.join(log_dir, 'results_all_metrics.csv'))
        logger.info(f"Results saved to {log_dir}/results_all_metrics.csv")

    # Published baselines for comparison
    logger.info("\n--- Published PerturBench Table 2 baselines (srivatsan20-transfer) ---")
    logger.info("Linear:     RMSE_rank=0.060, Cosine_rank=0.028")
    logger.info("LA:         RMSE_rank=0.059, Cosine_rank=0.032")
    logger.info("CPA:        RMSE_rank=0.085, Cosine_rank=0.049")
    logger.info("SAMS-VAE:   RMSE_rank=0.128, Cosine_rank=0.128")

    total_time = time.time() - t0
    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
