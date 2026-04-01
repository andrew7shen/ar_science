"""
Reproduce the predictive model of ALM inter-hemispheric coupling from
Chen, Kang et al. (2021) Cell 184:3717-3730.

Implements the full pipeline across all sessions:
  1. Load and preprocess bilateral ALM recordings
  2. Build choice decoder (CD) per hemisphere
  3. Compute top PCs orthogonal to CD
  4. Fit within-hemisphere autoregressive model
  5. Fit state-dependent cross-hemisphere coupling model
  6. Validate with scramble controls and state-dependent gating
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
DATA_PATH = Path(__file__).parent.parent / "brain_interaction_files" / "data"
MAT_FILE = DATA_PATH / "data_structure_BAYLORGC95_2019_12_19.mat"
OUT_DIR = Path(__file__).parent / "results"

# Model parameters (from STAR Methods)
BIN_WIDTH = 0.4       # 400ms windows
BIN_STEP = 0.1        # 100ms stride
T_START = -3.5        # seconds relative to go cue
T_END = 1.0
L_ANALYZED = 7        # self-lags (0..6)
L_CONTRA = 2          # contra lags (starts at -1, so lags -1,0,1)
N_CONTRA_COMPONENTS = 4  # CD + 3 PCs
LATE_DELAY_START = -0.9  # late delay epoch
LATE_DELAY_END = -0.2
N_FOLDS = 5
ALPHAS = np.logspace(-2, 6, 50)  # ridge regularization search


def get_session_files(session_files=None):
    """Get list of (session_name, mat_path) tuples."""
    if session_files is None:
        mat_files = sorted(DATA_PATH.glob("data_structure_*.mat"))
    else:
        mat_files = [DATA_PATH / f for f in session_files]
    return [(f.stem.replace("data_structure_", ""), f) for f in mat_files]


# ── Step 1: Load Data ──────────────────────────────────────────────────

def decode_string(f, ref):
    """Decode a uint16 HDF5 object reference to a Python string."""
    arr = f[ref]
    return "".join(chr(c) for c in arr[:, 0])


def load_session(mat_file):
    """Load all relevant data from the .mat HDF5 file."""
    f = h5py.File(mat_file, "r")
    obj = f["obj"]

    # Trial info
    trial_start_times = obj["trialStartTimes"][:, 0]
    trial_type_mat = obj["trialTypeMat"][:]  # (n_trials, 8)
    n_trials = len(trial_start_times)

    # Decode trial type names
    tts = obj["trialTypeStr"]
    trial_type_names = []
    for i in range(tts.shape[1]):
        trial_type_names.append(decode_string(f, tts[0, i]))

    # Trial properties
    tph_val = obj["trialPropertiesHash"]["value"]
    cue_time = f[tph_val[2, 0]][1, :]       # row 1 has actual cue times
    good_trials = f[tph_val[3, 0]][0, :]     # 0/1
    photostim_type = f[tph_val[4, 0]][0, :]  # 0=control

    # Build trial type boolean arrays
    type_idx = {name: i for i, name in enumerate(trial_type_names)}
    hitr = trial_type_mat[:, type_idx["HitR"]].astype(bool)
    hitl = trial_type_mat[:, type_idx["HitL"]].astype(bool)
    errr = trial_type_mat[:, type_idx["ErrR"]].astype(bool)
    errl = trial_type_mat[:, type_idx["ErrL"]].astype(bool)
    stim = trial_type_mat[:, type_idx["StimTrials"]].astype(bool)

    ctrl = photostim_type == 0
    good = good_trials == 1

    # Neuron data
    esh_val = obj["eventSeriesHash"]["value"]
    n_units = esh_val.shape[0]

    units = []
    for i in range(n_units):
        unit = f[esh_val[i, 0]]

        # cellType: object ref -> string, or uint64 -> "unclassified"
        ct_ds = unit["cellType"]
        if ct_ds.dtype == h5py.ref_dtype or str(ct_ds.dtype).startswith("|O"):
            ref = ct_ds[0, 0] if ct_ds.ndim == 2 else ct_ds[0]
            cell_type = decode_string(f, ref)
        else:
            cell_type = "unclassified"

        # hemisphere is always an object ref
        h_ds = unit["hemisphere"]
        ref = h_ds[0, 0] if h_ds.ndim == 2 else h_ds[0]
        hemisphere = decode_string(f, ref)

        quality = unit["manual_quality_score"][0, 0]
        spike_times = unit["eventTimes"][0, :]
        spike_trials = unit["eventTrials"][0, :].astype(int)
        units.append({
            "cell_type": cell_type,
            "hemisphere": hemisphere,
            "quality": quality,
            "spike_times": spike_times,
            "spike_trials": spike_trials,
        })

    return {
        "f": f,
        "n_trials": n_trials,
        "trial_start_times": trial_start_times,
        "cue_time": cue_time,
        "hitr": hitr, "hitl": hitl, "errr": errr, "errl": errl,
        "stim": stim, "ctrl": ctrl, "good": good,
        "units": units,
    }


# ── Step 2: Preprocess & Filter ───────────────────────────────────────

def filter_units(data):
    """Filter to pyramidal, quality=1 units. Separate by hemisphere."""
    left_idx, right_idx = [], []
    for i, u in enumerate(data["units"]):
        if u["cell_type"] == "pyramidal" and u["quality"] == 1.0:
            if u["hemisphere"] == "left ALM":
                left_idx.append(i)
            elif u["hemisphere"] == "right ALM":
                right_idx.append(i)
    return left_idx, right_idx


def get_trial_mask(data, trial_type):
    """Get boolean mask for control, good trials of given type."""
    ctrl, good = data["ctrl"], data["good"]
    if trial_type == "correct_right":
        return data["hitr"] & ctrl & good
    elif trial_type == "correct_left":
        return data["hitl"] & ctrl & good
    elif trial_type == "error_right":
        return data["errr"] & ctrl & good
    elif trial_type == "error_left":
        return data["errl"] & ctrl & good
    elif trial_type == "all_correct":
        return (data["hitr"] | data["hitl"]) & ctrl & good
    elif trial_type == "all_control_good":
        return ctrl & good & ~data["stim"]


# ── Step 3: Build Activity Matrix ─────────────────────────────────────

def build_activity_matrix(data, unit_indices, trial_mask):
    """
    Build firing rate matrix: (n_neurons, n_timebins, n_trials).
    Time bins are 400ms windows at 100ms steps, aligned to go cue.
    """
    bin_centers = np.arange(T_START + BIN_WIDTH / 2, T_END - BIN_WIDTH / 2 + BIN_STEP / 2, BIN_STEP)
    n_bins = len(bin_centers)
    trial_indices = np.where(trial_mask)[0]
    n_trials = len(trial_indices)
    n_neurons = len(unit_indices)

    activity = np.zeros((n_neurons, n_bins, n_trials))

    for ni, ui in enumerate(unit_indices):
        u = data["units"][ui]
        for ti_out, ti in enumerate(trial_indices):
            trial_num = ti + 1  # 1-indexed trial numbers in spike data
            # Get spikes for this trial, convert to cue-aligned time
            spike_mask = u["spike_trials"] == trial_num
            if not spike_mask.any():
                continue
            spk = u["spike_times"][spike_mask]
            # Convert to trial-relative time, then cue-relative
            spk_rel = spk - data["trial_start_times"][ti]
            spk_cue = spk_rel - data["cue_time"][ti]

            for bi, bc in enumerate(bin_centers):
                t0 = bc - BIN_WIDTH / 2
                t1 = bc + BIN_WIDTH / 2
                count = np.sum((spk_cue >= t0) & (spk_cue < t1))
                activity[ni, bi, ti_out] = count / BIN_WIDTH  # firing rate in Hz

    return activity, bin_centers


# ── Step 4: Choice Decoder (CD) ───────────────────────────────────────

def compute_cd(activity, trial_mask_r, trial_mask_l, bin_centers):
    """
    Compute choice decoder via vector difference of firing rates.
    Returns the fixed CD axis (averaged over delay epoch) and
    single-trial CD projections.
    """
    # Mean activity for right vs left correct trials
    mean_r = activity[:, :, :trial_mask_r.sum()].mean(axis=2)  # (n_neurons, n_bins)
    n_r = trial_mask_r.sum()
    mean_l = activity[:, :, n_r:n_r + trial_mask_l.sum()].mean(axis=2)

    # CD at each time point
    cd_per_bin = mean_r - mean_l  # (n_neurons, n_bins)

    # Average CD over delay epoch (-1.7s to -0.4s as in MATLAB code)
    delay_mask = (bin_centers >= -1.7) & (bin_centers <= -0.4)
    cd_axis = cd_per_bin[:, delay_mask].mean(axis=1)  # (n_neurons,)
    cd_axis = cd_axis / (np.linalg.norm(cd_axis) + 1e-10)  # normalize

    # Project all trials
    n_trials = activity.shape[2]
    cd_proj = np.zeros((n_trials, len(bin_centers)))
    for t in range(n_trials):
        cd_proj[t, :] = activity[:, :, t].T @ cd_axis

    return cd_axis, cd_proj


def compute_cd_v2(activity_all, right_indices, left_indices, bin_centers):
    """
    Compute CD from combined activity matrix with explicit trial indices.
    right_indices, left_indices index into axis=2 of activity_all.
    """
    mean_r = activity_all[:, :, right_indices].mean(axis=2)
    mean_l = activity_all[:, :, left_indices].mean(axis=2)

    cd_per_bin = mean_r - mean_l
    delay_mask = (bin_centers >= -1.7) & (bin_centers <= -0.4)
    cd_axis = cd_per_bin[:, delay_mask].mean(axis=1)
    cd_axis = cd_axis / (np.linalg.norm(cd_axis) + 1e-10)

    n_trials = activity_all.shape[2]
    cd_proj = np.zeros((n_trials, len(bin_centers)))
    for t in range(n_trials):
        cd_proj[t, :] = activity_all[:, :, t].T @ cd_axis

    return cd_axis, cd_proj


# ── Step 5: PCA Orthogonal to CD ──────────────────────────────────────

def compute_orthogonal_pcs(activity, cd_axis, bin_centers, n_pcs=3):
    """
    Compute top PCs of population activity, orthogonalized to CD.
    Returns single-trial projections onto each PC.
    """
    n_neurons, n_bins, n_trials = activity.shape

    # Flatten activity during delay for PCA: (n_samples, n_neurons)
    delay_mask = (bin_centers >= -1.7) & (bin_centers <= -0.4)
    delay_bins = np.where(delay_mask)[0]
    X_pca = []
    for t in range(n_trials):
        for b in delay_bins:
            X_pca.append(activity[:, b, t])
    X_pca = np.array(X_pca)  # (n_samples, n_neurons)

    # PCA
    pca = PCA(n_components=min(n_pcs + 5, n_neurons))
    pca.fit(X_pca)

    # Gram-Schmidt: orthogonalize PC axes to CD
    axes = []
    cd_norm = cd_axis / (np.linalg.norm(cd_axis) + 1e-10)
    for k in range(pca.n_components_):
        v = pca.components_[k].copy()
        # Remove CD component
        v = v - np.dot(v, cd_norm) * cd_norm
        # Remove components of previously accepted axes
        for a in axes:
            v = v - np.dot(v, a) * a
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm
            axes.append(v)
        if len(axes) == n_pcs:
            break

    # Project all trials onto each orthogonal PC
    pc_projs = np.zeros((n_pcs, n_trials, n_bins))
    for k, ax in enumerate(axes):
        for t in range(n_trials):
            pc_projs[k, t, :] = activity[:, :, t].T @ ax

    return axes, pc_projs


# ── Step 6: Construct Regression Features ─────────────────────────────

def build_regression_data(cd_proj_analyzed, cd_proj_contra, pc_projs_contra,
                          bin_centers, trial_types_per_trial):
    """
    Build features and targets for the coupling model.

    Target: V_t = cd_proj_analyzed[t+1] - cd_proj_analyzed[t]

    Self features: temporal differences of analyzed CD at lags 0..L_ANALYZED-1
      X_tilde_self[lag] = cd_proj_analyzed[t-lag] - cd_proj_analyzed[t-lag-1]

    Contra features: temporal differences of contra CD+PCs at lags -1..L_CONTRA-1
      X_tilde_contra[lag, comp] = contra_comp[t-lag] - contra_comp[t-lag-1]

    Restricted to late delay epoch.
    """
    n_trials, n_bins = cd_proj_analyzed.shape
    late_mask = (bin_centers >= LATE_DELAY_START) & (bin_centers <= LATE_DELAY_END)
    late_bins = np.where(late_mask)[0]

    # We need enough lags before and one step after
    max_lag_self = L_ANALYZED  # need t - L_ANALYZED
    min_lag_contra = -1  # contra starts at lag -1 (i.e. t+1)
    max_lag_contra = L_CONTRA - 1  # need t - (L_CONTRA-1)

    targets = []
    self_features = []
    contra_features = []
    state_labels = []  # per-sample state z_t
    trial_ids = []     # track which trial each sample comes from

    # Compute median CD per trial type for state classification
    # State = "highly selective" if |cd_proj| > median during late delay
    cd_late = cd_proj_analyzed[:, late_mask]
    median_by_type = {}
    for tt in ["right", "left"]:
        mask = trial_types_per_trial == tt
        if mask.any():
            median_by_type[tt] = np.median(np.abs(cd_late[mask, :]))
    # Fallback
    if not median_by_type:
        median_by_type["right"] = median_by_type["left"] = np.median(np.abs(cd_late))

    for trial in range(n_trials):
        tt = trial_types_per_trial[trial]
        med = median_by_type.get(tt, np.median(list(median_by_type.values())))

        for bi in late_bins:
            # Check we have enough lags
            if bi - max_lag_self - 1 < 0:
                continue
            if bi + 2 >= n_bins:  # need t+1 for target
                continue

            # Target: V_t = cd[t+1] - cd[t]
            v_t = cd_proj_analyzed[trial, bi + 1] - cd_proj_analyzed[trial, bi]
            targets.append(v_t)

            # Self features: temporal diffs at lags 0..L_ANALYZED-1
            sf = []
            for lag in range(L_ANALYZED):
                idx = bi - lag
                diff = cd_proj_analyzed[trial, idx] - cd_proj_analyzed[trial, idx - 1]
                sf.append(diff)
            self_features.append(sf)

            # Contra features: temporal diffs at lags -1..L_CONTRA-1 for all components
            cf = []
            # Component 0: contra CD
            for lag in range(-1, L_CONTRA):
                idx = bi - lag
                if idx < 1 or idx >= n_bins:
                    cf.append(0.0)
                else:
                    diff = cd_proj_contra[trial, idx] - cd_proj_contra[trial, idx - 1]
                    cf.append(diff)
            # Components 1..3: contra PCs
            for pc in range(pc_projs_contra.shape[0]):
                for lag in range(-1, L_CONTRA):
                    idx = bi - lag
                    if idx < 1 or idx >= n_bins:
                        cf.append(0.0)
                    else:
                        diff = pc_projs_contra[pc, trial, idx] - pc_projs_contra[pc, trial, idx - 1]
                        cf.append(diff)
            contra_features.append(cf)

            # State classification
            cd_val = cd_proj_analyzed[trial, bi]
            # "Highly selective" = far from boundary and on correct side
            if tt == "right":
                is_highly = cd_val > med
            else:
                is_highly = cd_val < -med
            state_labels.append(1 if is_highly else 0)
            trial_ids.append(trial)

    targets = np.array(targets)
    self_features = np.array(self_features)
    contra_features = np.array(contra_features)
    state_labels = np.array(state_labels)
    trial_ids = np.array(trial_ids)

    return targets, self_features, contra_features, state_labels, trial_ids


# ── Step 7: Fit Model via Ridge Regression ────────────────────────────

def fit_coupling_model(targets, self_features, contra_features, state_labels, trial_ids):
    """
    Two-stage fitting:
    1. Within-hemisphere: fit alpha on self_features only, 5-fold CV ridge
    2. Cross-hemisphere: fit beta on residuals, state-dependent, 5-fold CV ridge

    Returns R² values and beta coefficients.
    """
    n_samples = len(targets)
    unique_trials = np.unique(trial_ids)
    n_trials = len(unique_trials)

    # Create trial-level folds to avoid data leakage
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    trial_folds = list(kf.split(unique_trials))

    within_r2_folds = []
    full_r2_folds = []
    delta_r2_folds = []
    beta_highly_all = []
    beta_weakly_all = []

    # Also store for scramble control
    within_preds = np.zeros(n_samples)
    full_preds = np.zeros(n_samples)
    actual = targets.copy()

    for fold_i, (train_trial_idx, test_trial_idx) in enumerate(trial_folds):
        train_trials = set(unique_trials[train_trial_idx])
        test_trials = set(unique_trials[test_trial_idx])

        train_mask = np.array([tid in train_trials for tid in trial_ids])
        test_mask = np.array([tid in test_trials for tid in trial_ids])

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue

        # Stage 1: Within-hemisphere model
        X_self_train = self_features[train_mask]
        y_train = targets[train_mask]
        X_self_test = self_features[test_mask]
        y_test = targets[test_mask]

        ridge_within = RidgeCV(alphas=ALPHAS, cv=3)
        ridge_within.fit(X_self_train, y_train)

        within_pred_train = ridge_within.predict(X_self_train)
        within_pred_test = ridge_within.predict(X_self_test)
        within_preds[test_mask] = within_pred_test

        ss_res_within = np.sum((y_test - within_pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2_within = 1 - ss_res_within / ss_tot if ss_tot > 0 else 0
        within_r2_folds.append(r2_within)

        # Stage 2: Residuals
        residuals_train = y_train - within_pred_train
        residuals_test = y_test - within_pred_test

        # Split by state
        states_train = state_labels[train_mask]
        states_test = state_labels[test_mask]
        X_contra_train = contra_features[train_mask]
        X_contra_test = contra_features[test_mask]

        # Fit separate ridge for each state
        contra_pred_test = np.zeros(test_mask.sum())

        for state_val, label in [(1, "highly"), (0, "weakly")]:
            s_train = states_train == state_val
            s_test = states_test == state_val

            if s_train.sum() < 5:
                continue

            ridge_contra = RidgeCV(alphas=ALPHAS, cv=min(3, max(2, s_train.sum() // 3)))
            ridge_contra.fit(X_contra_train[s_train], residuals_train[s_train])

            if s_test.any():
                contra_pred_test[s_test] = ridge_contra.predict(X_contra_test[s_test])

            if label == "highly":
                beta_highly_all.append(ridge_contra.coef_.copy())
            else:
                beta_weakly_all.append(ridge_contra.coef_.copy())

        full_pred_test = within_pred_test + contra_pred_test
        full_preds[test_mask] = full_pred_test

        ss_res_full = np.sum((y_test - full_pred_test) ** 2)
        r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0
        full_r2_folds.append(r2_full)
        delta_r2_folds.append(r2_full - r2_within)

    return {
        "within_r2": within_r2_folds,
        "full_r2": full_r2_folds,
        "delta_r2": delta_r2_folds,
        "beta_highly": beta_highly_all,
        "beta_weakly": beta_weakly_all,
        "within_preds": within_preds,
        "full_preds": full_preds,
        "actual": actual,
    }


def scramble_control(targets, self_features, contra_features, state_labels,
                     trial_ids, n_permutations=50):
    """
    Scramble trial identity of contra hemisphere features.
    Returns distribution of delta R² under null hypothesis.
    """
    unique_trials = np.unique(trial_ids)
    delta_r2_null = []

    for perm in range(n_permutations):
        # Shuffle contra features at trial level
        rng = np.random.RandomState(perm)
        perm_map = dict(zip(unique_trials, rng.permutation(unique_trials)))
        scrambled_contra = np.zeros_like(contra_features)
        for i, tid in enumerate(trial_ids):
            new_tid = perm_map[tid]
            donor_mask = trial_ids == new_tid
            donor_indices = np.where(donor_mask)[0]
            if len(donor_indices) > 0:
                # Pick a random sample from the donor trial
                j = donor_indices[rng.randint(len(donor_indices))]
                scrambled_contra[i] = contra_features[j]

        result = fit_coupling_model(targets, self_features, scrambled_contra,
                                    state_labels, trial_ids)
        if result["delta_r2"]:
            delta_r2_null.append(np.mean(result["delta_r2"]))

    return delta_r2_null


# ── Step 8: Validation & Plotting ─────────────────────────────────────

def compute_decoder_accuracy(cd_proj, right_mask_local, left_mask_local, bin_centers):
    """Compute time-resolved decoding accuracy from CD projections."""
    delay_mask = (bin_centers >= -1.7) & (bin_centers <= -0.4)
    delay_bins = np.where(delay_mask)[0]

    # Average CD during delay per trial
    cd_delay = cd_proj[:, delay_bins].mean(axis=1)

    # Simple threshold decoder: positive -> right, negative -> left
    n_r = right_mask_local.sum()
    n_l = left_mask_local.sum()
    correct = 0
    total = 0
    for i in range(len(cd_delay)):
        if right_mask_local[i]:
            correct += cd_delay[i] > 0
            total += 1
        elif left_mask_local[i]:
            correct += cd_delay[i] < 0
            total += 1
    return correct / total if total > 0 else 0.5


def plot_cd_trajectories(cd_proj, bin_centers, right_mask, left_mask, hemisphere,
                         full_preds, within_preds, actual, trial_ids, save_path):
    """Plot actual vs predicted CD change trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: mean CD trajectories by trial type
    ax = axes[0]
    mean_r = cd_proj[right_mask, :].mean(axis=0)
    mean_l = cd_proj[left_mask, :].mean(axis=0)
    sem_r = cd_proj[right_mask, :].std(axis=0) / np.sqrt(right_mask.sum())
    sem_l = cd_proj[left_mask, :].std(axis=0) / np.sqrt(left_mask.sum())

    ax.plot(bin_centers, mean_r, "r-", label="Lick Right")
    ax.fill_between(bin_centers, mean_r - sem_r, mean_r + sem_r, color="r", alpha=0.2)
    ax.plot(bin_centers, mean_l, "b-", label="Lick Left")
    ax.fill_between(bin_centers, mean_l - sem_l, mean_l + sem_l, color="b", alpha=0.2)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("CD projection (Hz)")
    ax.set_title(f"{hemisphere} CD trajectories")
    ax.legend()

    # Right panel: actual vs predicted delta-CD
    ax = axes[1]
    ax.scatter(actual, full_preds, alpha=0.3, s=10, label="Full model")
    ax.scatter(actual, within_preds, alpha=0.2, s=10, label="Within only")
    lim = max(abs(actual.max()), abs(actual.min()), abs(full_preds.max()), abs(full_preds.min()))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
    ax.set_xlabel("Actual ΔCD")
    ax.set_ylabel("Predicted ΔCD")
    ax.set_title(f"{hemisphere}: Predicted vs Actual")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_delta_r2(real_delta_r2, scrambled_delta_r2, hemisphere, save_path):
    """Plot real vs scrambled ΔR²."""
    fig, ax = plt.subplots(figsize=(6, 4))

    if scrambled_delta_r2:
        ax.hist(scrambled_delta_r2, bins=20, color="gray", alpha=0.7, label="Scrambled")
    ax.axvline(np.mean(real_delta_r2), color="red", lw=2, label=f"Real: {np.mean(real_delta_r2):.4f}")
    ax.set_xlabel("ΔR² (contra improvement)")
    ax.set_ylabel("Count")
    ax.set_title(f"{hemisphere}: ΔR² real vs scrambled")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_beta_by_state(beta_highly, beta_weakly, hemisphere, save_path):
    """Plot contra beta coefficients for highly vs weakly selective states."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if beta_highly:
        bh = np.mean(beta_highly, axis=0)
        ax.bar(np.arange(len(bh)) - 0.15, np.abs(bh), width=0.3,
               label="Highly selective", color="steelblue", alpha=0.8)
    if beta_weakly:
        bw = np.mean(beta_weakly, axis=0)
        ax.bar(np.arange(len(bw)) + 0.15, np.abs(bw), width=0.3,
               label="Weakly selective", color="coral", alpha=0.8)

    # Label groups
    n_lags = L_CONTRA + 1  # lags -1..L_CONTRA-1
    labels = []
    for comp_name in ["CD", "PC1", "PC2", "PC3"]:
        for lag in range(-1, L_CONTRA):
            labels.append(f"{comp_name}\nlag {lag}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Contra feature")
    ax.set_ylabel("|β|")
    ax.set_title(f"{hemisphere}: Contra β by state (Q2: gating)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Single Session Pipeline ───────────────────────────────────────────

def run_single_session(mat_path, session_name, plots_dir, log):
    """Run the full coupling model pipeline on one session. Returns results dict."""
    session_plots = plots_dir / session_name
    session_plots.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    log("[Step 1] Loading data...")
    data = load_session(mat_path)
    log(f"  Trials: {data['n_trials']}")
    log(f"  Units: {len(data['units'])}")

    # Step 2: Filter
    log("[Step 2] Filtering units...")
    left_idx, right_idx = filter_units(data)
    log(f"  Left ALM pyramidal (quality=1): {len(left_idx)}")
    log(f"  Right ALM pyramidal (quality=1): {len(right_idx)}")

    mask_cr = get_trial_mask(data, "correct_right")
    mask_cl = get_trial_mask(data, "correct_left")
    mask_all_correct = get_trial_mask(data, "all_correct")
    log(f"  Correct right (control, good): {mask_cr.sum()}")
    log(f"  Correct left (control, good): {mask_cl.sum()}")

    # Step 3: Build activity matrices
    log("[Step 3] Building activity matrices...")
    correct_trials = np.where(mask_all_correct)[0]
    right_correct_trials = np.where(mask_cr)[0]
    left_correct_trials = np.where(mask_cl)[0]

    activity_left, bin_centers = build_activity_matrix(data, left_idx, mask_all_correct)
    activity_right, _ = build_activity_matrix(data, right_idx, mask_all_correct)
    n_correct = mask_all_correct.sum()
    log(f"  Activity shape (left): {activity_left.shape}")
    log(f"  Activity shape (right): {activity_right.shape}")
    log(f"  Time bins: {len(bin_centers)} ({bin_centers[0]:.2f}s to {bin_centers[-1]:.2f}s)")

    correct_trial_nums = correct_trials
    right_local = np.array([i for i, t in enumerate(correct_trial_nums) if t in right_correct_trials])
    left_local = np.array([i for i, t in enumerate(correct_trial_nums) if t in left_correct_trials])

    right_mask_local = np.zeros(n_correct, dtype=bool)
    right_mask_local[right_local] = True
    left_mask_local = np.zeros(n_correct, dtype=bool)
    left_mask_local[left_local] = True

    # Step 4: CD per hemisphere
    log("[Step 4] Computing choice decoders...")
    cd_axis_left, cd_proj_left = compute_cd_v2(activity_left, right_local, left_local, bin_centers)
    cd_axis_right, cd_proj_right = compute_cd_v2(activity_right, right_local, left_local, bin_centers)

    acc_left = compute_decoder_accuracy(cd_proj_left, right_mask_local, left_mask_local, bin_centers)
    acc_right = compute_decoder_accuracy(cd_proj_right, right_mask_local, left_mask_local, bin_centers)
    log(f"  Left ALM decoder accuracy: {acc_left:.1%}")
    log(f"  Right ALM decoder accuracy: {acc_right:.1%}")

    # Step 5: PCA orthogonal to CD
    log("[Step 5] Computing orthogonal PCs...")
    _, pc_projs_left = compute_orthogonal_pcs(activity_left, cd_axis_left, bin_centers, n_pcs=3)
    _, pc_projs_right = compute_orthogonal_pcs(activity_right, cd_axis_right, bin_centers, n_pcs=3)
    log(f"  PC projections shape (left): {pc_projs_left.shape}")

    trial_types = np.array(["right" if right_mask_local[i] else "left" for i in range(n_correct)])

    results_all = {}

    for analyzed_hemi, contra_hemi in [("left", "right"), ("right", "left")]:
        log("=" * 60)
        log(f"Analyzing: {analyzed_hemi} ALM (contra = {contra_hemi} ALM)")
        log("=" * 60)

        if analyzed_hemi == "left":
            cd_analyzed, cd_contra, pc_contra = cd_proj_left, cd_proj_right, pc_projs_right
        else:
            cd_analyzed, cd_contra, pc_contra = cd_proj_right, cd_proj_left, pc_projs_left

        # Step 6: Build regression data
        log("[Step 6] Building regression features...")
        targets, self_feat, contra_feat, states, tids = build_regression_data(
            cd_analyzed, cd_contra, pc_contra, bin_centers, trial_types
        )
        log(f"  Samples: {len(targets)}")
        log(f"  Self features: {self_feat.shape[1]}")
        log(f"  Contra features: {contra_feat.shape[1]}")
        log(f"  Highly selective: {(states == 1).sum()}, Weakly: {(states == 0).sum()}")

        # Step 7: Fit model
        log("[Step 7] Fitting coupling model (5-fold CV)...")
        result = fit_coupling_model(targets, self_feat, contra_feat, states, tids)

        mean_within_r2 = np.mean(result["within_r2"])
        mean_full_r2 = np.mean(result["full_r2"])
        mean_delta_r2 = np.mean(result["delta_r2"])
        log(f"  Within-hemisphere R²: {mean_within_r2:.4f} (±{np.std(result['within_r2']):.4f})")
        log(f"  Full model R²:       {mean_full_r2:.4f} (±{np.std(result['full_r2']):.4f})")
        log(f"  ΔR² (contra):        {mean_delta_r2:.4f} (±{np.std(result['delta_r2']):.4f})")

        if result["beta_highly"] and result["beta_weakly"]:
            bh_mag = np.mean([np.mean(np.abs(b)) for b in result["beta_highly"]])
            bw_mag = np.mean([np.mean(np.abs(b)) for b in result["beta_weakly"]])
            log(f"  Mean |β| highly selective: {bh_mag:.4f}")
            log(f"  Mean |β| weakly selective: {bw_mag:.4f}")
            log(f"  Gating (weakly > highly): {bw_mag > bh_mag}")
        else:
            bh_mag = bw_mag = None

        # Scramble control
        log("[Step 7b] Running scramble control (50 permutations)...")
        scrambled = scramble_control(targets, self_feat, contra_feat, states, tids, n_permutations=50)
        if scrambled:
            p_scramble = np.mean([s >= mean_delta_r2 for s in scrambled])
            log(f"  Scrambled ΔR² mean: {np.mean(scrambled):.4f}")
            log(f"  p-value (real > scrambled): {p_scramble:.3f}")
        else:
            p_scramble = None

        results_all[analyzed_hemi] = {
            "decoder_accuracy": acc_left if analyzed_hemi == "left" else acc_right,
            "within_r2_mean": float(mean_within_r2),
            "within_r2_std": float(np.std(result["within_r2"])),
            "full_r2_mean": float(mean_full_r2),
            "full_r2_std": float(np.std(result["full_r2"])),
            "delta_r2_mean": float(mean_delta_r2),
            "delta_r2_std": float(np.std(result["delta_r2"])),
            "delta_r2_folds": [float(x) for x in result["delta_r2"]],
            "beta_highly_mean_abs": float(bh_mag) if bh_mag else None,
            "beta_weakly_mean_abs": float(bw_mag) if bw_mag else None,
            "gating_correct_direction": bool(bw_mag > bh_mag) if bh_mag and bw_mag else None,
            "scrambled_delta_r2_mean": float(np.mean(scrambled)) if scrambled else None,
            "scramble_pvalue": float(p_scramble) if p_scramble is not None else None,
        }

        # Plots
        log("[Step 8] Generating plots...")
        plot_cd_trajectories(
            cd_analyzed, bin_centers, right_mask_local, left_mask_local,
            f"{analyzed_hemi.title()} ALM",
            result["full_preds"], result["within_preds"], result["actual"],
            tids,
            session_plots / f"cd_trajectories_{analyzed_hemi}.png"
        )
        plot_delta_r2(
            result["delta_r2"], scrambled, f"{analyzed_hemi.title()} ALM",
            session_plots / f"delta_r2_vs_scrambled_{analyzed_hemi}.png"
        )
        plot_beta_by_state(
            result["beta_highly"], result["beta_weakly"],
            f"{analyzed_hemi.title()} ALM",
            session_plots / f"beta_by_state_{analyzed_hemi}.png"
        )

    # Per-session summary
    log("-" * 40)
    log("SESSION SUMMARY")
    for hemi in ["left", "right"]:
        r = results_all[hemi]
        log(f"  {hemi.upper()} ALM: decoder={r['decoder_accuracy']:.1%}, "
            f"ΔR²={r['delta_r2_mean']:.4f}, "
            f"gating={r.get('gating_correct_direction')}, "
            f"p={r.get('scramble_pvalue')}")

    data["f"].close()
    return results_all


# ── Cross-Session Summary ─────────────────────────────────────────────

def compute_cross_session_summary(all_results):
    """Compute aggregate metrics across successful sessions."""
    summary = {}
    successful = {k: v for k, v in all_results.items() if "error" not in v}
    n_ok = len(successful)

    for hemi in ["left", "right"]:
        vals = {}
        for metric in ["delta_r2_mean", "within_r2_mean", "full_r2_mean", "decoder_accuracy"]:
            arr = [s[hemi][metric] for s in successful.values()
                   if hemi in s and s[hemi].get(metric) is not None]
            if arr:
                vals[f"{metric}_across_sessions"] = float(np.mean(arr))
                vals[f"{metric}_std_across_sessions"] = float(np.std(arr))
                vals[f"{metric}_median_across_sessions"] = float(np.median(arr))

        gating = [s[hemi]["gating_correct_direction"] for s in successful.values()
                   if hemi in s and s[hemi].get("gating_correct_direction") is not None]
        vals["gating_correct_fraction"] = float(np.mean(gating)) if gating else None

        pvals = [s[hemi]["scramble_pvalue"] for s in successful.values()
                 if hemi in s and s[hemi].get("scramble_pvalue") is not None]
        vals["scramble_sig_fraction"] = float(np.mean([p < 0.05 for p in pvals])) if pvals else None

        summary[hemi] = vals

    summary["n_sessions"] = n_ok
    summary["n_failed"] = len(all_results) - n_ok
    return summary


def log_cross_session_summary(all_results, summary, log):
    """Log a table of cross-session results."""
    log("")
    log("=" * 70)
    log("CROSS-SESSION SUMMARY")
    log("=" * 70)

    successful = {k: v for k, v in all_results.items() if "error" not in v}

    # Per-session table
    log(f"  {'Session':<35} {'L ΔR²':>8} {'R ΔR²':>8} {'L gate':>7} {'R gate':>7}")
    log(f"  {'-'*65}")
    for name, res in sorted(successful.items()):
        l = res.get("left", {})
        r = res.get("right", {})
        l_dr2 = f"{l.get('delta_r2_mean', 0):.4f}" if l else "N/A"
        r_dr2 = f"{r.get('delta_r2_mean', 0):.4f}" if r else "N/A"
        l_gate = str(l.get("gating_correct_direction", "N/A")) if l else "N/A"
        r_gate = str(r.get("gating_correct_direction", "N/A")) if r else "N/A"
        log(f"  {name:<35} {l_dr2:>8} {r_dr2:>8} {l_gate:>7} {r_gate:>7}")

    # Aggregate
    log("")
    for hemi in ["left", "right"]:
        s = summary[hemi]
        log(f"  {hemi.upper()} ALM aggregate ({summary['n_sessions']} sessions):")
        dr2 = s.get("delta_r2_mean_across_sessions")
        dr2_std = s.get("delta_r2_mean_std_across_sessions")
        if dr2 is not None:
            log(f"    Mean ΔR²: {dr2:.4f} (±{dr2_std:.4f})")
        gf = s.get("gating_correct_fraction")
        if gf is not None:
            log(f"    Gating correct fraction: {gf:.1%}")
        sf = s.get("scramble_sig_fraction")
        if sf is not None:
            log(f"    Scramble p<0.05 fraction: {sf:.1%}")

    failed = [(k, v["error"]) for k, v in all_results.items() if "error" in v]
    if failed:
        log(f"\n  FAILED SESSIONS ({len(failed)}):")
        for name, err in failed:
            log(f"    {name}: {err}")


def plot_cross_session_summary(all_results, summary_dir):
    """Bar chart of delta_r2 per session for left and right ALM."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    successful = {k: v for k, v in all_results.items() if "error" not in v}
    if not successful:
        return

    sessions = sorted(successful.keys())
    left_dr2 = [successful[s].get("left", {}).get("delta_r2_mean", 0) for s in sessions]
    right_dr2 = [successful[s].get("right", {}).get("delta_r2_mean", 0) for s in sessions]

    fig, ax = plt.subplots(figsize=(max(10, len(sessions) * 0.7), 5))
    x = np.arange(len(sessions))
    w = 0.35
    ax.bar(x - w/2, left_dr2, w, label="Left ALM", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, right_dr2, w, label="Right ALM", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_", 1)[-1] if "_" in s else s for s in sessions],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ΔR² (contra improvement)")
    ax.set_title("Coupling Model: ΔR² Across Sessions")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(summary_dir / "delta_r2_across_sessions.png", dpi=150)
    plt.close()


# ── Main Pipeline (Multi-Session Orchestrator) ───────────────────────

def run_pipeline(session_files=None):
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUT_DIR / f"coupling_{run_timestamp}"
    plots_dir = run_dir / "plots"
    summary_plots_dir = plots_dir / "summary"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('coupling_model')
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
    log("Reproducing Chen et al. 2021 coupling model")
    log(f"Processing {len(sessions)} sessions")
    log(f"Output: {run_dir}")
    log("=" * 60)

    all_results = {}
    failed_sessions = []

    for i, (session_name, mat_path) in enumerate(sessions):
        log(f"\n{'#' * 60}")
        log(f"[Session {i+1}/{len(sessions)}] {session_name}")
        log(f"{'#' * 60}")
        try:
            result = run_single_session(mat_path, session_name, plots_dir, log)
            all_results[session_name] = result
        except Exception as e:
            log(f"  SESSION FAILED: {e}")
            failed_sessions.append((session_name, str(e)))
            all_results[session_name] = {"error": str(e)}

    # Cross-session summary
    summary = compute_cross_session_summary(all_results)
    log_cross_session_summary(all_results, summary, log)
    plot_cross_session_summary(all_results, summary_plots_dir)

    # Save unified JSON
    output = {
        "sessions": all_results,
        "summary": summary,
        "failed_sessions": failed_sessions,
        "n_sessions_total": len(sessions),
        "n_sessions_succeeded": len(sessions) - len(failed_sessions),
    }
    results_path = run_dir / "coupling_model_results.json"
    with open(results_path, "w") as fout:
        json.dump(output, fout, indent=2)
    log(f"\nResults saved to {results_path}")
    log(f"Plots saved to {plots_dir}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coupling model baseline")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Specific .mat filenames. Default: all sessions.")
    args = parser.parse_args()
    run_pipeline(session_files=args.sessions)
