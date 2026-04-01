"""
Native PCMCI Causal Discovery for ALM Inter-Hemispheric Dynamics.

Uses PCMCI natively: raw time series (no differencing), partial correlation
outputs (no Ridge R²). PCMCI handles autocorrelation via conditional
independence testing, directly isolating cross-hemisphere influence.

Metrics: significant link counts, link density, mean/max/sum partial
correlations, asymmetry, lag distributions.

Permutation test: shuffle cross-hemisphere trial pairings and re-run PCMCI.
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import shared steps from coupling model
sys.path.insert(0, str(Path(__file__).parent.parent / "coupling_model_implementation"))
from reproduce_coupling_model import (
    load_session, filter_units, get_trial_mask, build_activity_matrix,
    compute_cd_v2, compute_orthogonal_pcs, compute_decoder_accuracy,
    get_session_files,
    BIN_WIDTH, BIN_STEP, T_START, T_END,
    LATE_DELAY_START, LATE_DELAY_END,
)

# Paths
DATA_PATH = Path(__file__).parent.parent / "brain_interaction_files" / "data"
OUT_DIR = Path(__file__).parent / "results"

# PCMCI parameters
TAU_MAX = 3
PC_ALPHA = 0.05
VAR_NAMES = ["CD_L", "PC1_L", "PC2_L", "PC3_L", "CD_R", "PC1_R", "PC2_R", "PC3_R"]
N_LEFT = 4   # indices 0-3 are left hemisphere
N_RIGHT = 4  # indices 4-7 are right hemisphere
N_VARS = N_LEFT + N_RIGHT

# Max possible cross-hemisphere links (each direction): 4 sources * 4 targets * (TAU_MAX+1) lags
# But tau=0 same-var excluded only for auto-links; cross-hemi tau=0 is valid
MAX_CROSS_LINKS = N_LEFT * N_RIGHT * (TAU_MAX + 1)


# ── Data Building ─────────────────────────────────────────────────────

def build_pcmci_data(cd_proj_left, cd_proj_right, pc_projs_left, pc_projs_right,
                     trial_mask, bin_centers):
    """
    Build PCMCI input: 8 variables x T time bins per trial, multi-dataset mode.
    Raw data (no differencing) — PCMCI handles autocorrelation natively via ParCorr.
    """
    trial_indices = np.where(trial_mask)[0] if trial_mask.dtype == bool else trial_mask
    data_dict = {}
    for ti_idx, ti in enumerate(trial_indices):
        trial_data = np.column_stack([
            cd_proj_left[ti, :],
            pc_projs_left[0, ti, :],
            pc_projs_left[1, ti, :],
            pc_projs_left[2, ti, :],
            cd_proj_right[ti, :],
            pc_projs_right[0, ti, :],
            pc_projs_right[1, ti, :],
            pc_projs_right[2, ti, :],
        ])
        data_dict[ti_idx] = trial_data
    return data_dict


# ── PCMCI Analysis (reused) ──────────────────────────────────────────

def run_pcmci_analysis(data_dict, log=None):
    """Run PCMCI on multi-dataset DataFrame. Returns results dict."""
    df = pp.DataFrame(data_dict, analysis_mode='multiple', var_names=VAR_NAMES)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=df, cond_ind_test=parcorr, verbosity=0)

    if log:
        log(f"  Running PCMCI (tau_max={TAU_MAX}, pc_alpha={PC_ALPHA})...")
        log(f"  Datasets: {len(data_dict)}, time steps per dataset: "
            f"{next(iter(data_dict.values())).shape[0]}")

    results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    return results, pcmci


# ── Native Metrics Extraction (NEW) ──────────────────────────────────

def extract_native_metrics(results, alpha=0.05):
    """
    Extract all native coupling metrics from PCMCI results.

    Returns dict with:
    - r2l/l2r link counts, densities, mean/max/sum partial correlations
    - asymmetry metrics
    - autoregressive link count
    - lag distribution of cross-hemisphere links
    """
    val_matrix = results['val_matrix']
    p_matrix = results['p_matrix']

    r2l_links = []  # right (4-7) → left (0-3)
    l2r_links = []  # left (0-3) → right (4-7)
    auto_links = []

    for target in range(N_VARS):
        for source in range(N_VARS):
            for tau in range(TAU_MAX + 1):
                if tau == 0 and source == target:
                    continue
                p_val = p_matrix[source, target, tau]
                val = val_matrix[source, target, tau]
                if p_val >= alpha:
                    continue

                link = {
                    "source": VAR_NAMES[source], "source_idx": int(source),
                    "target": VAR_NAMES[target], "target_idx": int(target),
                    "tau": int(tau),
                    "val": float(val), "p": float(p_val),
                }

                # Classify link
                if source >= N_LEFT and target < N_LEFT:
                    r2l_links.append(link)
                elif source < N_LEFT and target >= N_LEFT:
                    l2r_links.append(link)
                elif source == target and tau > 0:
                    auto_links.append(link)

    r2l_vals = np.array([l["val"] for l in r2l_links]) if r2l_links else np.array([])
    l2r_vals = np.array([l["val"] for l in l2r_links]) if l2r_links else np.array([])

    r2l_count = len(r2l_links)
    l2r_count = len(l2r_links)

    # Lag distribution for R→L links
    r2l_lag_dist = {}
    for tau in range(TAU_MAX + 1):
        r2l_lag_dist[tau] = sum(1 for l in r2l_links if l["tau"] == tau)

    metrics = {
        # Link counts
        "r2l_link_count": r2l_count,
        "l2r_link_count": l2r_count,
        # Link densities (fraction of possible cross-links)
        "r2l_link_density": r2l_count / MAX_CROSS_LINKS if MAX_CROSS_LINKS > 0 else 0.0,
        "l2r_link_density": l2r_count / MAX_CROSS_LINKS if MAX_CROSS_LINKS > 0 else 0.0,
        # Partial correlation stats
        "r2l_mean_abs_parcorr": float(np.mean(np.abs(r2l_vals))) if len(r2l_vals) > 0 else 0.0,
        "l2r_mean_abs_parcorr": float(np.mean(np.abs(l2r_vals))) if len(l2r_vals) > 0 else 0.0,
        "r2l_max_abs_parcorr": float(np.max(np.abs(r2l_vals))) if len(r2l_vals) > 0 else 0.0,
        "r2l_sum_abs_parcorr": float(np.sum(np.abs(r2l_vals))) if len(r2l_vals) > 0 else 0.0,
        "l2r_max_abs_parcorr": float(np.max(np.abs(l2r_vals))) if len(l2r_vals) > 0 else 0.0,
        "l2r_sum_abs_parcorr": float(np.sum(np.abs(l2r_vals))) if len(l2r_vals) > 0 else 0.0,
        # Asymmetry
        "asymmetry_count": r2l_count - l2r_count,
        "asymmetry_strength": (
            float(np.mean(np.abs(r2l_vals))) - float(np.mean(np.abs(l2r_vals)))
            if len(r2l_vals) > 0 and len(l2r_vals) > 0
            else float(np.mean(np.abs(r2l_vals))) if len(r2l_vals) > 0
            else -float(np.mean(np.abs(l2r_vals))) if len(l2r_vals) > 0
            else 0.0
        ),
        "r2l_dominant": r2l_count > l2r_count,
        # Autoregressive sanity check
        "auto_link_count": len(auto_links),
        # Lag distribution
        "r2l_lag_distribution": r2l_lag_dist,
        # Raw link lists for detailed inspection
        "r2l_links": r2l_links,
        "l2r_links": l2r_links,
    }
    return metrics


# ── Trial Classification (reused) ────────────────────────────────────

def classify_trials_by_selectivity(cd_proj, right_mask_local, left_mask_local, bin_centers):
    """
    Split trials into weakly/highly selective based on median |CD| during late delay.
    Returns boolean masks for weakly and highly selective trials.
    """
    late_mask = (bin_centers >= LATE_DELAY_START) & (bin_centers <= LATE_DELAY_END)
    cd_late = cd_proj[:, late_mask]
    n_trials = cd_proj.shape[0]

    mean_abs_cd = np.mean(np.abs(cd_late), axis=1)

    weakly_mask = np.zeros(n_trials, dtype=bool)
    highly_mask = np.zeros(n_trials, dtype=bool)

    for type_mask in [right_mask_local, left_mask_local]:
        if type_mask.sum() == 0:
            continue
        type_indices = np.where(type_mask)[0]
        median_cd = np.median(mean_abs_cd[type_indices])
        for idx in type_indices:
            if mean_abs_cd[idx] <= median_cd:
                weakly_mask[idx] = True
            else:
                highly_mask[idx] = True

    return weakly_mask, highly_mask


# ── Permutation Test (NEW) ───────────────────────────────────────────

def permutation_test_native(cd_proj_left, cd_proj_right, pc_projs_left, pc_projs_right,
                            trial_indices, bin_centers, n_permutations=20, log=None):
    """
    Shuffle cross-hemisphere trial pairings and re-run PCMCI.

    For each permutation:
    1. Keep left hemisphere trials in original order
    2. Shuffle which right hemisphere trial is paired with each left trial
    3. Run PCMCI on the shuffled data
    4. Extract native metrics

    Returns list of null metrics dicts.
    """
    null_metrics = []

    for perm in range(n_permutations):
        if log:
            log(f"    Permutation {perm + 1}/{n_permutations}...")

        rng = np.random.RandomState(perm)
        shuffled_right_order = rng.permutation(len(trial_indices))

        # Build shuffled data: left stays, right gets shuffled trial pairing
        data_dict = {}
        for ti_idx, ti in enumerate(trial_indices):
            shuffled_right_ti = trial_indices[shuffled_right_order[ti_idx]]
            trial_data = np.column_stack([
                cd_proj_left[ti, :],
                pc_projs_left[0, ti, :],
                pc_projs_left[1, ti, :],
                pc_projs_left[2, ti, :],
                cd_proj_right[shuffled_right_ti, :],
                pc_projs_right[0, shuffled_right_ti, :],
                pc_projs_right[1, shuffled_right_ti, :],
                pc_projs_right[2, shuffled_right_ti, :],
            ])
            data_dict[ti_idx] = trial_data

        results, _ = run_pcmci_analysis(data_dict)
        metrics = extract_native_metrics(results)
        null_metrics.append(metrics)

    return null_metrics


def compute_permutation_pvalues(real_metrics, null_metrics):
    """
    Compute p-values: fraction of null permutations >= real observed value.
    Tests key metrics: link counts, mean partial correlations.
    """
    test_keys = [
        "r2l_link_count", "l2r_link_count",
        "r2l_mean_abs_parcorr", "l2r_mean_abs_parcorr",
        "r2l_sum_abs_parcorr", "l2r_sum_abs_parcorr",
        "asymmetry_count",
    ]
    pvalues = {}
    for key in test_keys:
        real_val = real_metrics[key]
        null_vals = [m[key] for m in null_metrics]
        pvalues[key] = float(np.mean([n >= real_val for n in null_vals]))
        pvalues[f"{key}_null_mean"] = float(np.mean(null_vals))
        pvalues[f"{key}_null_std"] = float(np.std(null_vals))

    return pvalues


# ── Plotting Functions ────────────────────────────────────────────────

def plot_causal_graph(results, pcmci, title, save_path, alpha=0.05):
    """Plot PCMCI causal graph using tigramite's plotting."""
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        tp.plot_graph(
            val_matrix=results['val_matrix'],
            graph=results['graph'],
            var_names=VAR_NAMES,
            fig_ax=(fig, ax),
            link_colorbar_label='Partial correlation',
            node_colorbar_label='Auto-dependency',
        )
        ax.set_title(title, fontsize=14)
    except Exception:
        ax.clear()
        _plot_adjacency_fallback(results, ax, title, alpha)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_adjacency_fallback(results, ax, title, alpha):
    """Fallback adjacency heatmap if tigramite plotting fails."""
    val_matrix = results['val_matrix']
    p_matrix = results['p_matrix']

    adj = np.zeros((N_VARS, N_VARS))
    for i in range(N_VARS):
        for j in range(N_VARS):
            vals = []
            for tau in range(val_matrix.shape[2]):
                if p_matrix[i, j, tau] < alpha:
                    vals.append(val_matrix[i, j, tau])
            if vals:
                adj[i, j] = vals[np.argmax(np.abs(vals))]

    im = ax.imshow(adj, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(N_VARS))
    ax.set_yticks(range(N_VARS))
    ax.set_xticklabels(VAR_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(VAR_NAMES, fontsize=8)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Partial correlation")

    for i in range(N_VARS):
        for j in range(N_VARS):
            if abs(adj[i, j]) > 0:
                ax.text(j, i, f"{adj[i,j]:.2f}", ha='center', va='center', fontsize=6)


def plot_link_heatmap(metrics, title, save_path):
    """
    Source x target-lag heatmap showing partial correlations for cross-hemisphere links.
    Two panels: R→L and L→R.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, direction, links in [
        (axes[0], "Right → Left", metrics["r2l_links"]),
        (axes[1], "Left → Right", metrics["l2r_links"]),
    ]:
        if direction == "Right → Left":
            source_names = VAR_NAMES[N_LEFT:]
            target_names = VAR_NAMES[:N_LEFT]
            source_offset = N_LEFT
            target_offset = 0
        else:
            source_names = VAR_NAMES[:N_LEFT]
            target_names = VAR_NAMES[N_LEFT:]
            source_offset = 0
            target_offset = N_LEFT

        # Build heatmap: rows = source, cols = target*lag
        n_sources = len(source_names)
        n_targets = len(target_names)
        n_lags = TAU_MAX + 1
        heatmap = np.full((n_sources, n_targets * n_lags), np.nan)

        for link in links:
            s_idx = link["source_idx"] - source_offset
            t_idx = link["target_idx"] - target_offset
            tau = link["tau"]
            col = t_idx * n_lags + tau
            heatmap[s_idx, col] = link["val"]

        # Column labels: "CD_L(0)", "CD_L(1)", ...
        col_labels = []
        for t_name in target_names:
            for tau in range(n_lags):
                col_labels.append(f"{t_name}(τ={tau})")

        im = ax.imshow(heatmap, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n_sources))
        ax.set_yticklabels(source_names, fontsize=8)
        ax.set_xlabel("Target (lag)")
        ax.set_ylabel("Source")
        ax.set_title(f"{direction} ({len(links)} sig. links)")
        plt.colorbar(im, ax=ax, label="Partial corr.", shrink=0.8)

        # Mark significant links
        for link in links:
            s_idx = link["source_idx"] - source_offset
            t_idx = link["target_idx"] - target_offset
            tau = link["tau"]
            col = t_idx * n_lags + tau
            ax.text(col, s_idx, f"{link['val']:.2f}", ha='center', va='center',
                    fontsize=5, color='white' if abs(link['val']) > 0.15 else 'black')

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_coupling_asymmetry(metrics, title, save_path):
    """Bar chart: R→L vs L→R link counts and mean strengths."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Link counts
    ax = axes[0]
    counts = [metrics["r2l_link_count"], metrics["l2r_link_count"]]
    colors = ["coral", "steelblue"]
    bars = ax.bar(["R→L", "L→R"], counts, color=colors, alpha=0.8)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', va='bottom', fontsize=11)
    ax.set_ylabel("# significant links")
    ax.set_title("Cross-Hemisphere Link Counts")

    # Mean strengths
    ax = axes[1]
    strengths = [metrics["r2l_mean_abs_parcorr"], metrics["l2r_mean_abs_parcorr"]]
    bars = ax.bar(["R→L", "L→R"], strengths, color=colors, alpha=0.8)
    for bar, v in zip(bars, strengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Mean |partial correlation|")
    ax.set_title("Cross-Hemisphere Link Strength")

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gating_comparison_native(weakly_metrics, highly_metrics, title, save_path):
    """Weakly vs highly selective comparison of R→L coupling (native metrics)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    conditions = ["Weakly sel.", "Highly sel."]
    colors = ["coral", "steelblue"]

    # Link counts
    ax = axes[0]
    vals = [weakly_metrics["r2l_link_count"], highly_metrics["r2l_link_count"]]
    bars = ax.bar(conditions, vals, color=colors, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("# R→L links")
    ax.set_title("R→L Link Count")

    # Mean strength
    ax = axes[1]
    vals = [weakly_metrics["r2l_mean_abs_parcorr"], highly_metrics["r2l_mean_abs_parcorr"]]
    bars = ax.bar(conditions, vals, color=colors, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Mean |partial corr.|")
    ax.set_title("R→L Mean Strength")

    # Link density
    ax = axes[2]
    vals = [weakly_metrics["r2l_link_density"], highly_metrics["r2l_link_density"]]
    bars = ax.bar(conditions, vals, color=colors, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Link density")
    ax.set_title("R→L Link Density")

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_permutation_null(real_metrics, perm_pvalues, null_metrics, title, save_path):
    """Null distribution histograms for key metrics with real value marked."""
    test_keys = ["r2l_link_count", "r2l_mean_abs_parcorr", "r2l_sum_abs_parcorr",
                 "asymmetry_count"]
    labels = ["R→L Link Count", "R→L Mean |PartCorr|", "R→L Sum |PartCorr|",
              "Asymmetry (R→L − L→R)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, key, label in zip(axes, test_keys, labels):
        null_vals = [m[key] for m in null_metrics]
        real_val = real_metrics[key]
        p_val = perm_pvalues[key]

        ax.hist(null_vals, bins=15, color='gray', alpha=0.7, edgecolor='black', label='Null')
        ax.axvline(real_val, color='red', lw=2, ls='--', label=f'Real = {real_val:.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\np = {p_val:.3f}")
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_session_summary_native(all_results, summary_dir):
    """Cross-session bar charts for native PCMCI metrics."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    successful = {k: v for k, v in all_results.items() if "error" not in v}
    if not successful:
        return

    sessions = sorted(successful.keys())

    # Plot 1: R→L vs L→R link counts across sessions
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(sessions) * 0.7), 9))

    ax = axes[0]
    r2l_counts = [successful[s]["native_metrics"]["r2l_link_count"] for s in sessions]
    l2r_counts = [successful[s]["native_metrics"]["l2r_link_count"] for s in sessions]
    x = np.arange(len(sessions))
    w = 0.35
    ax.bar(x - w/2, r2l_counts, w, label="R→L", color="coral", alpha=0.8)
    ax.bar(x + w/2, l2r_counts, w, label="L→R", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_", 1)[-1] if "_" in s else s for s in sessions],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("# significant links")
    ax.set_title("Cross-Hemisphere Link Counts Across Sessions")
    ax.legend()

    # Plot 2: R→L mean strength across sessions
    ax = axes[1]
    r2l_strength = [successful[s]["native_metrics"]["r2l_mean_abs_parcorr"] for s in sessions]
    l2r_strength = [successful[s]["native_metrics"]["l2r_mean_abs_parcorr"] for s in sessions]
    ax.bar(x - w/2, r2l_strength, w, label="R→L", color="coral", alpha=0.8)
    ax.bar(x + w/2, l2r_strength, w, label="L→R", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_", 1)[-1] if "_" in s else s for s in sessions],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean |partial correlation|")
    ax.set_title("Cross-Hemisphere Link Strength Across Sessions")
    ax.legend()

    plt.tight_layout()
    plt.savefig(summary_dir / "native_metrics_across_sessions.png", dpi=150)
    plt.close()

    # Plot 3: Gating direction across sessions
    fig, ax = plt.subplots(figsize=(max(10, len(sessions) * 0.7), 5))
    weakly_r2l = [successful[s]["weakly_metrics"]["r2l_mean_abs_parcorr"] for s in sessions]
    highly_r2l = [successful[s]["highly_metrics"]["r2l_mean_abs_parcorr"] for s in sessions]
    ax.bar(x - w/2, weakly_r2l, w, label="Weakly sel.", color="coral", alpha=0.8)
    ax.bar(x + w/2, highly_r2l, w, label="Highly sel.", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_", 1)[-1] if "_" in s else s for s in sessions],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R→L Mean |partial correlation|")
    ax.set_title("State-Dependent R→L Coupling Across Sessions (Q2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(summary_dir / "gating_across_sessions.png", dpi=150)
    plt.close()


# ── Single Session Pipeline (NEW) ────────────────────────────────────

def run_single_session(mat_path, session_name, plots_dir, log, n_permutations=20):
    """Run native PCMCI pipeline on one session. Returns results dict."""
    session_plots = plots_dir / session_name
    session_plots.mkdir(parents=True, exist_ok=True)

    # ── Steps 1-5: Reuse coupling model pipeline ────────────────────
    log("[Step 1] Loading data...")
    data = load_session(mat_path)
    log(f"  Trials: {data['n_trials']}, Units: {len(data['units'])}")

    log("[Step 2] Filtering units...")
    left_idx, right_idx = filter_units(data)
    log(f"  Left ALM: {len(left_idx)}, Right ALM: {len(right_idx)}")

    mask_cr = get_trial_mask(data, "correct_right")
    mask_cl = get_trial_mask(data, "correct_left")
    mask_all_correct = get_trial_mask(data, "all_correct")
    log(f"  Correct right: {mask_cr.sum()}, Correct left: {mask_cl.sum()}")

    log("[Step 3] Building activity matrices...")
    correct_trials = np.where(mask_all_correct)[0]
    right_correct_trials = np.where(mask_cr)[0]
    left_correct_trials = np.where(mask_cl)[0]

    activity_left, bin_centers = build_activity_matrix(data, left_idx, mask_all_correct)
    activity_right, _ = build_activity_matrix(data, right_idx, mask_all_correct)
    n_correct = mask_all_correct.sum()
    log(f"  Activity left: {activity_left.shape}, right: {activity_right.shape}")
    log(f"  Time bins: {len(bin_centers)}")

    correct_trial_nums = correct_trials
    right_local = np.array([i for i, t in enumerate(correct_trial_nums) if t in right_correct_trials])
    left_local = np.array([i for i, t in enumerate(correct_trial_nums) if t in left_correct_trials])

    right_mask_local = np.zeros(n_correct, dtype=bool)
    right_mask_local[right_local] = True
    left_mask_local = np.zeros(n_correct, dtype=bool)
    left_mask_local[left_local] = True

    log("[Step 4] Computing choice decoders...")
    cd_axis_left, cd_proj_left = compute_cd_v2(activity_left, right_local, left_local, bin_centers)
    cd_axis_right, cd_proj_right = compute_cd_v2(activity_right, right_local, left_local, bin_centers)

    acc_left = compute_decoder_accuracy(cd_proj_left, right_mask_local, left_mask_local, bin_centers)
    acc_right = compute_decoder_accuracy(cd_proj_right, right_mask_local, left_mask_local, bin_centers)
    log(f"  Decoder accuracy: left={acc_left:.1%}, right={acc_right:.1%}")

    log("[Step 5] Computing orthogonal PCs...")
    _, pc_projs_left = compute_orthogonal_pcs(activity_left, cd_axis_left, bin_centers, n_pcs=3)
    _, pc_projs_right = compute_orthogonal_pcs(activity_right, cd_axis_right, bin_centers, n_pcs=3)

    # ── Step 6: Build PCMCI data (raw, no differencing) ─────────────
    log("[Step 6] Building PCMCI input data (raw, no differencing)...")
    all_trial_indices = np.arange(n_correct)
    data_all = build_pcmci_data(cd_proj_left, cd_proj_right, pc_projs_left, pc_projs_right,
                                all_trial_indices, bin_centers)
    log(f"  All trials: {len(data_all)} datasets")
    log(f"  Shape per trial: {next(iter(data_all.values())).shape}")

    # Step 6b: Classify trials by selectivity
    weakly_mask, highly_mask = classify_trials_by_selectivity(
        cd_proj_left, right_mask_local, left_mask_local, bin_centers)
    log(f"  Weakly selective: {weakly_mask.sum()}, Highly selective: {highly_mask.sum()}")

    weakly_indices = np.where(weakly_mask)[0]
    highly_indices = np.where(highly_mask)[0]

    data_weakly = build_pcmci_data(cd_proj_left, cd_proj_right, pc_projs_left, pc_projs_right,
                                   weakly_indices, bin_centers)
    data_highly = build_pcmci_data(cd_proj_left, cd_proj_right, pc_projs_left, pc_projs_right,
                                   highly_indices, bin_centers)

    # ── Step 7: Run PCMCI on raw data ─────────────────────────────
    log("=" * 60)
    log("[Step 7] PCMCI on RAW time series (native mode)")
    log("")
    log("[Step 7a] PCMCI on ALL trials (Q1: detect coupling)...")
    results_all, pcmci_all = run_pcmci_analysis(data_all, log=log)
    all_metrics = extract_native_metrics(results_all)

    log(f"  R→L significant links: {all_metrics['r2l_link_count']}")
    log(f"  L→R significant links: {all_metrics['l2r_link_count']}")
    log(f"  R→L mean |parcorr|: {all_metrics['r2l_mean_abs_parcorr']:.4f}")
    log(f"  L→R mean |parcorr|: {all_metrics['l2r_mean_abs_parcorr']:.4f}")
    log(f"  R→L max |parcorr|: {all_metrics['r2l_max_abs_parcorr']:.4f}")
    log(f"  R→L dominant: {all_metrics['r2l_dominant']}")
    log(f"  Autoregressive links: {all_metrics['auto_link_count']} (of {N_VARS * TAU_MAX} possible)")
    log(f"  R→L lag distribution: {all_metrics['r2l_lag_distribution']}")

    log("")
    log("[Step 7b] PCMCI on WEAKLY selective trials...")
    results_weakly, pcmci_weakly = run_pcmci_analysis(data_weakly, log=log)
    weakly_metrics = extract_native_metrics(results_weakly)
    log(f"  R→L links: {weakly_metrics['r2l_link_count']}, "
        f"mean |parcorr|: {weakly_metrics['r2l_mean_abs_parcorr']:.4f}")

    log("")
    log("[Step 7c] PCMCI on HIGHLY selective trials...")
    results_highly, pcmci_highly = run_pcmci_analysis(data_highly, log=log)
    highly_metrics = extract_native_metrics(results_highly)
    log(f"  R→L links: {highly_metrics['r2l_link_count']}, "
        f"mean |parcorr|: {highly_metrics['r2l_mean_abs_parcorr']:.4f}")

    # Q2 gating
    gating_weakly_stronger = (weakly_metrics['r2l_mean_abs_parcorr'] >
                              highly_metrics['r2l_mean_abs_parcorr'])
    log(f"  Q2 gating: weakly stronger = {gating_weakly_stronger}")

    # ── Step 8: Permutation test ──────────────────────────────────
    log("")
    log(f"[Step 8] Permutation test ({n_permutations} permutations)...")
    null_metrics = permutation_test_native(
        cd_proj_left, cd_proj_right, pc_projs_left, pc_projs_right,
        all_trial_indices, bin_centers, n_permutations=n_permutations, log=log)

    perm_pvalues = compute_permutation_pvalues(all_metrics, null_metrics)
    log(f"  R→L link count: real={all_metrics['r2l_link_count']}, "
        f"null mean={perm_pvalues['r2l_link_count_null_mean']:.1f}, "
        f"p={perm_pvalues['r2l_link_count']:.3f}")
    log(f"  R→L mean |parcorr|: real={all_metrics['r2l_mean_abs_parcorr']:.4f}, "
        f"null mean={perm_pvalues['r2l_mean_abs_parcorr_null_mean']:.4f}, "
        f"p={perm_pvalues['r2l_mean_abs_parcorr']:.3f}")
    log(f"  Asymmetry count: real={all_metrics['asymmetry_count']}, "
        f"null mean={perm_pvalues['asymmetry_count_null_mean']:.1f}, "
        f"p={perm_pvalues['asymmetry_count']:.3f}")

    # ── Step 9: Plots ─────────────────────────────────────────────
    log("")
    log("[Step 9] Generating plots...")

    plot_causal_graph(results_all, pcmci_all, f"Native PCMCI - All Trials ({session_name})",
                      session_plots / "causal_graph_all.png")
    plot_causal_graph(results_weakly, pcmci_weakly,
                      f"Native PCMCI - Weakly Sel. ({session_name})",
                      session_plots / "causal_graph_weakly.png")
    plot_causal_graph(results_highly, pcmci_highly,
                      f"Native PCMCI - Highly Sel. ({session_name})",
                      session_plots / "causal_graph_highly.png")

    plot_link_heatmap(all_metrics, f"Link Heatmap ({session_name})",
                      session_plots / "link_heatmap.png")
    plot_coupling_asymmetry(all_metrics, f"Coupling Asymmetry ({session_name})",
                            session_plots / "coupling_asymmetry.png")
    plot_gating_comparison_native(weakly_metrics, highly_metrics,
                                 f"State-Dependent Coupling ({session_name})",
                                 session_plots / "gating_comparison.png")
    plot_permutation_null(all_metrics, perm_pvalues, null_metrics,
                          f"Permutation Test ({session_name})",
                          session_plots / "permutation_null.png")

    # ── Build output ──────────────────────────────────────────────
    # Strip link lists for JSON (keep them compact)
    def compact_metrics(m):
        return {k: v for k, v in m.items() if k not in ("r2l_links", "l2r_links")}

    output = {
        "decoder_accuracy_left": float(acc_left),
        "decoder_accuracy_right": float(acc_right),
        "native_metrics": compact_metrics(all_metrics),
        "weakly_metrics": compact_metrics(weakly_metrics),
        "highly_metrics": compact_metrics(highly_metrics),
        "gating_weakly_stronger": gating_weakly_stronger,
        "r2l_dominant": all_metrics["r2l_dominant"],
        "permutation_pvalues": perm_pvalues,
    }

    data["f"].close()
    return output


# ── Cross-Session Summary (NEW) ──────────────────────────────────────

def compute_cross_session_summary(all_results):
    """Aggregate native PCMCI metrics across sessions."""
    successful = {k: v for k, v in all_results.items() if "error" not in v}
    n_ok = len(successful)

    # Aggregate key metrics
    metric_keys = [
        "r2l_link_count", "l2r_link_count",
        "r2l_link_density", "l2r_link_density",
        "r2l_mean_abs_parcorr", "l2r_mean_abs_parcorr",
        "r2l_max_abs_parcorr", "r2l_sum_abs_parcorr",
        "asymmetry_count", "auto_link_count",
    ]

    aggregated = {}
    for key in metric_keys:
        vals = [s["native_metrics"][key] for s in successful.values()]
        aggregated[f"{key}_mean"] = float(np.mean(vals))
        aggregated[f"{key}_std"] = float(np.std(vals))
        aggregated[f"{key}_median"] = float(np.median(vals))

    # Q1: fraction with R→L dominant
    r2l_dominant = [s["r2l_dominant"] for s in successful.values()]
    aggregated["r2l_dominant_fraction"] = float(np.mean(r2l_dominant))

    # Q2: fraction with gating in correct direction
    gating = [s["gating_weakly_stronger"] for s in successful.values()]
    aggregated["gating_correct_fraction"] = float(np.mean(gating))

    # Permutation test: fraction significant
    r2l_pvals = [s["permutation_pvalues"]["r2l_link_count"] for s in successful.values()]
    aggregated["r2l_count_sig_fraction"] = float(np.mean([p < 0.05 for p in r2l_pvals]))

    summary = {
        "n_sessions": n_ok,
        "n_failed": len(all_results) - n_ok,
        "aggregated_metrics": aggregated,
    }
    return summary


def log_cross_session_summary(all_results, summary, log):
    """Log cross-session summary table."""
    log("")
    log("=" * 70)
    log("CROSS-SESSION SUMMARY (Native PCMCI)")
    log("=" * 70)

    successful = {k: v for k, v in all_results.items() if "error" not in v}
    agg = summary["aggregated_metrics"]

    log(f"  {'Session':<35} {'R→L#':>5} {'L→R#':>5} {'R→L|ρ|':>8} {'Asym':>6} {'Gate':>6} {'Perm p':>7}")
    log(f"  {'-'*72}")
    for name, res in sorted(successful.items()):
        m = res["native_metrics"]
        p = res["permutation_pvalues"]["r2l_link_count"]
        log(f"  {name:<35} {m['r2l_link_count']:>5} {m['l2r_link_count']:>5} "
            f"{m['r2l_mean_abs_parcorr']:>8.4f} "
            f"{str(res['r2l_dominant']):>6} {str(res['gating_weakly_stronger']):>6} "
            f"{p:>7.3f}")

    log("")
    log(f"  Aggregate ({summary['n_sessions']} sessions):")
    log(f"    R→L link count: {agg['r2l_link_count_mean']:.1f} (±{agg['r2l_link_count_std']:.1f})")
    log(f"    L→R link count: {agg['l2r_link_count_mean']:.1f} (±{agg['l2r_link_count_std']:.1f})")
    log(f"    R→L mean |parcorr|: {agg['r2l_mean_abs_parcorr_mean']:.4f} (±{agg['r2l_mean_abs_parcorr_std']:.4f})")
    log(f"    Auto-regressive links: {agg['auto_link_count_mean']:.1f} (±{agg['auto_link_count_std']:.1f})")
    log(f"    R→L dominant fraction: {agg['r2l_dominant_fraction']:.1%}")
    log(f"    Gating correct fraction: {agg['gating_correct_fraction']:.1%}")
    log(f"    R→L permutation sig fraction: {agg['r2l_count_sig_fraction']:.1%}")

    failed = [(k, v["error"]) for k, v in all_results.items() if "error" in v]
    if failed:
        log(f"\n  FAILED SESSIONS ({len(failed)}):")
        for name, err in failed:
            log(f"    {name}: {err}")


# ── Main Pipeline ────────────────────────────────────────────────────

def run_pipeline(session_files=None, n_permutations=20):
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUT_DIR / f"native_pcmci_{run_timestamp}"
    plots_dir = run_dir / "plots"
    summary_plots_dir = plots_dir / "summary"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('pcmci_native')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / 'run.log', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    log = logger.info

    sessions = get_session_files(session_files)

    log("=" * 60)
    log("Native PCMCI Causal Discovery")
    log("Raw time series, partial correlation metrics, no Ridge R²")
    log(f"Processing {len(sessions)} sessions")
    log(f"Permutations: {n_permutations}")
    log(f"Output: {run_dir}")
    log("=" * 60)

    all_results = {}
    failed_sessions = []

    for i, (session_name, mat_path) in enumerate(sessions):
        log(f"\n{'#' * 60}")
        log(f"[Session {i+1}/{len(sessions)}] {session_name}")
        log(f"{'#' * 60}")
        try:
            result = run_single_session(mat_path, session_name, plots_dir, log,
                                        n_permutations=n_permutations)
            all_results[session_name] = result
        except Exception as e:
            log(f"  SESSION FAILED: {e}")
            import traceback
            log(traceback.format_exc())
            failed_sessions.append((session_name, str(e)))
            all_results[session_name] = {"error": str(e)}

    # Cross-session summary
    summary = compute_cross_session_summary(all_results)
    log_cross_session_summary(all_results, summary, log)
    plot_cross_session_summary_native(all_results, summary_plots_dir)

    # Save unified JSON
    output = {
        "sessions": all_results,
        "summary": summary,
        "failed_sessions": failed_sessions,
        "n_sessions_total": len(sessions),
        "n_sessions_succeeded": len(sessions) - len(failed_sessions),
        "parameters": {
            "tau_max": TAU_MAX,
            "pc_alpha": PC_ALPHA,
            "variables": VAR_NAMES,
            "n_permutations": n_permutations,
            "mode": "native (raw data, no differencing, no Ridge R²)",
        },
    }
    results_path = run_dir / "native_pcmci_results.json"
    with open(results_path, "w") as fout:
        json.dump(output, fout, indent=2)
    log(f"\nResults saved to {results_path}")
    log(f"Plots saved to {plots_dir}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native PCMCI causal discovery")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Specific .mat filenames. Default: all sessions.")
    parser.add_argument("--n-permutations", type=int, default=20,
                        help="Number of permutations for null distribution (default: 20)")
    args = parser.parse_args()
    run_pipeline(session_files=args.sessions, n_permutations=args.n_permutations)
