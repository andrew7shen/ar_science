"""
Piece-Square Tables (PST) with Tapered Evaluation for oligonucleotide activity prediction.

Analogous reasoning from chess: each monomer type at each position contributes differently
to functional activity, like piece-square tables in chess engines. Tapered evaluation blends
terminal-regime and central-regime weights, analogous to opening/endgame PSTs.

Evaluation follows the same protocol as reproduce_rf.py: 5-fold random CV + 5-fold nucleobase CV,
reporting PCC as the primary metric.
"""

import argparse
import itertools
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from oligogym.data import DatasetDownloader
from oligogym.features import OneHotEncoder
from oligogym.metrics import regression_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

# RF baseline results for comparison
RF_BASELINE = {
    "random": {"pcc_mean": 0.49, "pcc_std": 0.08},
    "nucleobase": {"pcc_mean": 0.40, "pcc_std": 0.14},
}

FEATURE_VARIANTS = ["PST-base", "PST-full", "PST-tapered"]

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

RF_GRID = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [10, 20, 30],
}

STRAND_OPTIONS = [None, ["RNA1"]]


def setup_logging(run_dir):
    logger = logging.getLogger("pst_eval")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "run.log", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger.info


def generate_configs():
    """Generate all PST configs: 15 Ridge + 54 RF = 69 total."""
    configs = []

    # Ridge configs: 3 feature variants × 5 alphas = 15
    for variant in FEATURE_VARIANTS:
        for alpha in RIDGE_ALPHAS:
            configs.append({
                "model_type": "Ridge",
                "feature_variant": variant,
                "strands": None,
                "model_args": {"alpha": alpha},
                "label": f"Ridge_{variant}_a{alpha}",
            })

    # RF configs: 3 feature variants × 2 strand options × 3 n_est × 3 max_depth = 54
    for variant, strands, (n_est, max_d) in itertools.product(
        FEATURE_VARIANTS,
        STRAND_OPTIONS,
        itertools.product(RF_GRID["n_estimators"], RF_GRID["max_depth"]),
    ):
        strand_label = "all" if strands is None else "RNA1"
        configs.append({
            "model_type": "RF",
            "feature_variant": variant,
            "strands": strands,
            "model_args": {"n_estimators": n_est, "max_depth": None},
            "max_depth_label": max_d,
            "label": f"RF_{variant}_{strand_label}_n{n_est}_d{max_d}",
        })

    return configs


def get_encoder(variant, strands=None):
    """Create OneHotEncoder for a given PST feature variant."""
    if variant == "PST-base":
        components = ["base"]
    else:  # PST-full and PST-tapered both use all components
        components = ["phosphate", "sugar", "base"]

    return OneHotEncoder(encode_components=components, strands=strands)


def apply_taper(X_3d):
    """Apply tapered evaluation: duplicate features with terminal/central weights.

    Args:
        X_3d: shape (n_samples, seq_len, vocab_size)
    Returns:
        X_tapered: shape (n_samples, 2 * seq_len * vocab_size)
    """
    n_samples, seq_len, vocab_size = X_3d.shape
    center = (seq_len - 1) / 2.0
    half_len = seq_len / 2.0

    # Terminal weight: high at ends, low at center
    positions = np.arange(seq_len)
    terminal_w = np.abs(positions - center) / half_len  # shape (seq_len,)
    central_w = 1.0 - terminal_w

    # Apply weights: broadcast over (n_samples, seq_len, vocab_size)
    X_terminal = X_3d * terminal_w[np.newaxis, :, np.newaxis]
    X_central = X_3d * central_w[np.newaxis, :, np.newaxis]

    # Flatten and concatenate
    X_terminal_flat = X_terminal.reshape(n_samples, -1)
    X_central_flat = X_central.reshape(n_samples, -1)
    return np.hstack([X_terminal_flat, X_central_flat])


def featurize(X_train, X_test, variant, strands=None):
    """Featurize using PST (position-specific one-hot) features."""
    encoder = get_encoder(variant, strands)
    X_train_3d = encoder.fit_transform(X_train)
    X_test_3d = encoder.transform(X_test)

    if variant == "PST-tapered":
        return apply_taper(X_train_3d), apply_taper(X_test_3d)
    else:
        n_train = X_train_3d.shape[0]
        n_test = X_test_3d.shape[0]
        return X_train_3d.reshape(n_train, -1), X_test_3d.reshape(n_test, -1)


def prepare_data_fold(data, k):
    """Replicate train_model.py's prepare_data_fold exactly (same as reproduce_rf.py)."""
    X, y = data.x, data.y
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // 5
    test_indices = indices[k * fold_size : (k + 1) * fold_size]
    train_indices = np.concatenate(
        [indices[: k * fold_size], indices[(k + 1) * fold_size :]]
    )
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def run_single_config(data, cv_strategy, config, seed):
    """Run 5-fold CV for one config. Returns list of metric dicts."""
    variant = config["feature_variant"]
    strands = config["strands"]
    model_type = config["model_type"]
    model_args = config["model_args"]

    np.random.seed(seed)
    results = []

    for k in range(5):
        if cv_strategy == "random":
            X_train, X_test, y_train, y_test = prepare_data_fold(data, k)
        else:
            data_fold = DatasetDownloader().download("Shmushkovich")
            X_train, X_test, y_train, y_test, _, _ = data_fold.split(
                "nucleobase", return_index=True
            )

        X_train_feat, X_test_feat = featurize(X_train, X_test, variant, strands)

        if model_type == "Ridge":
            model = Ridge(**model_args)
        else:
            model = RandomForestRegressor(random_state=seed + k, **model_args)

        model.fit(X_train_feat, y_train.ravel())
        y_pred = model.predict(X_test_feat)

        metrics = regression_metrics(y_test.squeeze(), y_pred.squeeze())
        metrics["fold"] = k
        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed; each config gets base_seed + config_index")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"pst_shmushkovich_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir)

    configs = generate_configs()
    log("=" * 60)
    log("PST with Tapered Evaluation on Shmushkovich (OligoGym)")
    log(f"Sweeping {len(configs)} configs (15 Ridge + 54 RF)")
    log(f"Base seed: {args.base_seed}")
    log(f"Output: {run_dir}")
    log("=" * 60)

    log("Downloading Shmushkovich dataset...")
    data = DatasetDownloader().download("Shmushkovich")
    log(f"Dataset size: {len(data.x)} samples")

    for cv in ["random", "nucleobase"]:
        log("")
        log(f"{'=' * 60}")
        log(f"{cv.upper()} CV: sweeping {len(configs)} configs")
        log(f"{'=' * 60}")

        all_rows = []
        for i, config in enumerate(configs):
            seed = args.base_seed + i
            fold_results = run_single_config(data, cv, config, seed)
            mean_pcc = np.mean([r["pearson_correlation"] for r in fold_results])

            for r in fold_results:
                r["cv_strategy"] = cv
                r["seed"] = seed
                r["config_id"] = i
                r["model_type"] = config["model_type"]
                r["feature_variant"] = config["feature_variant"]
                r["strands"] = str(config["strands"])
                r["model_args"] = str(config["model_args"])
                r["label"] = config["label"]
                all_rows.append(r)

            log(f"  Config {i:2d}: {config['label']:40s} seed={seed:3d} -> PCC={mean_pcc:.4f}")

        # Save all per-fold results
        df = pd.DataFrame(all_rows)
        df.to_csv(run_dir / f"all_configs_{cv}.csv", index=False)

        # Rank configs by mean PCC
        grouped = df.groupby("config_id").agg(
            pcc_mean=("pearson_correlation", "mean"),
            pcc_std=("pearson_correlation", "std"),
            r2_mean=("r2_score", "mean"),
            rmse_mean=("root_mean_squared_error", "mean"),
            mae_mean=("mean_absolute_error", "mean"),
            spearman_mean=("spearman_correlation", "mean"),
            label=("label", "first"),
            model_type=("model_type", "first"),
            feature_variant=("feature_variant", "first"),
            model_args=("model_args", "first"),
            seed=("seed", "first"),
        ).sort_values("pcc_mean", ascending=False)

        grouped.to_csv(run_dir / f"ranked_configs_{cv}.csv")

        best = grouped.iloc[0]
        baseline = RF_BASELINE[cv]

        log("")
        log(f"  BEST CONFIG ({cv} CV):")
        log(f"    Config:  {best['label']}")
        log(f"    Ours:    PCC = {best['pcc_mean']:.4f} +/- {best['pcc_std']:.4f}")
        log(f"    RF base: PCC = {baseline['pcc_mean']:.2f} +/- {baseline['pcc_std']:.2f}")
        diff = best["pcc_mean"] - baseline["pcc_mean"]
        log(f"    Delta:   {diff:+.4f}")

        # Show top 5
        log("")
        log(f"  TOP 5 CONFIGS ({cv} CV):")
        for rank, (_, row) in enumerate(grouped.head(5).iterrows()):
            log(f"    #{rank+1}: {row['label']:40s} PCC={row['pcc_mean']:.4f} +/- {row['pcc_std']:.4f}")

    log("")
    log(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
