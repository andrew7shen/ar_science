"""
PST + KMersCounts combined features with Linear model.

Matches the paper's Linear model settings exactly:
- Model types: 'standard' (LinearRegression) and 'ridge' (Ridge, alpha=1.0)
- Featurizer grid: same KMersCounts configs as paper
- Adds PST-only and PST+KMer combined configs using the same model types

This allows direct apples-to-apples comparison with the paper's Linear results.
"""

import argparse
import itertools
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from oligogym.data import DatasetDownloader
from oligogym.features import OneHotEncoder, KMersCounts
from oligogym.metrics import regression_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

PAPER_LINEAR = {
    "random": {"pcc_mean": 0.24, "pcc_std": 0.09},
    "nucleobase": {"pcc_mean": 0.20, "pcc_std": 0.08},
}

# Match paper's featurizer grid exactly
KMER_CONFIGS = [
    {"k": [1], "modification_abundance": False},
    {"k": [1], "modification_abundance": True},
    {"k": [1, 2], "modification_abundance": False},
    {"k": [1, 2], "modification_abundance": True},
    {"k": [1, 2, 3], "modification_abundance": False},
    {"k": [1, 2, 3], "modification_abundance": True},
]

PST_VARIANTS = ["PST-base", "PST-full", "PST-tapered"]

# Match paper's model types exactly: standard (L2=0) and ridge (L2=1)
MODEL_TYPES = ["standard", "ridge"]


def setup_logging(run_dir):
    logger = logging.getLogger("pst_kmer_eval")
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
    """Generate configs matching paper's Linear grid, plus PST variants."""
    configs = []

    # PST-only: 3 variants × 2 model types = 6
    for variant, model_type in itertools.product(PST_VARIANTS, MODEL_TYPES):
        configs.append({
            "pst_variant": variant,
            "kmer_args": None,
            "model_type": model_type,
            "label": f"{variant}_{model_type}",
        })

    # KMer-only (paper's Linear baseline): 6 kmer configs × 2 model types = 12
    for kmer_args, model_type in itertools.product(KMER_CONFIGS, MODEL_TYPES):
        k_str = "".join(map(str, kmer_args["k"]))
        mod = "mod" if kmer_args["modification_abundance"] else "nomod"
        configs.append({
            "pst_variant": None,
            "kmer_args": kmer_args,
            "model_type": model_type,
            "label": f"KMer-k{k_str}_{mod}_{model_type}",
        })

    # PST + KMer combined: 3 × 6 × 2 = 36
    for variant, kmer_args, model_type in itertools.product(
        PST_VARIANTS, KMER_CONFIGS, MODEL_TYPES
    ):
        k_str = "".join(map(str, kmer_args["k"]))
        mod = "mod" if kmer_args["modification_abundance"] else "nomod"
        configs.append({
            "pst_variant": variant,
            "kmer_args": kmer_args,
            "model_type": model_type,
            "label": f"{variant}+KMer-k{k_str}_{mod}_{model_type}",
        })

    return configs


def get_pst_encoder(variant):
    if variant == "PST-base":
        return OneHotEncoder(encode_components=["base"])
    else:
        return OneHotEncoder(encode_components=["phosphate", "sugar", "base"])


def apply_taper(X_3d):
    n_samples, seq_len, vocab_size = X_3d.shape
    center = (seq_len - 1) / 2.0
    half_len = seq_len / 2.0
    positions = np.arange(seq_len)
    terminal_w = np.abs(positions - center) / half_len
    central_w = 1.0 - terminal_w
    X_terminal = X_3d * terminal_w[np.newaxis, :, np.newaxis]
    X_central = X_3d * central_w[np.newaxis, :, np.newaxis]
    return np.hstack([X_terminal.reshape(n_samples, -1), X_central.reshape(n_samples, -1)])


def featurize(X_train, X_test, pst_variant, kmer_args):
    parts_train, parts_test = [], []

    if pst_variant is not None:
        encoder = get_pst_encoder(pst_variant)
        X_train_3d = encoder.fit_transform(X_train)
        X_test_3d = encoder.transform(X_test)
        if pst_variant == "PST-tapered":
            parts_train.append(apply_taper(X_train_3d))
            parts_test.append(apply_taper(X_test_3d))
        else:
            parts_train.append(X_train_3d.reshape(X_train_3d.shape[0], -1))
            parts_test.append(X_test_3d.reshape(X_test_3d.shape[0], -1))

    if kmer_args is not None:
        kmer = KMersCounts(**kmer_args)
        X_train_k = kmer.fit_transform(X_train)
        X_test_k = kmer.transform(X_test)
        if len(X_train_k.shape) == 3:
            X_train_k = X_train_k.reshape(X_train_k.shape[0], -1)
            X_test_k = X_test_k.reshape(X_test_k.shape[0], -1)
        parts_train.append(X_train_k)
        parts_test.append(X_test_k)

    return np.hstack(parts_train), np.hstack(parts_test)


def prepare_data_fold(data, k):
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


def make_model(model_type):
    if model_type == "standard":
        return LinearRegression()
    else:  # ridge
        return Ridge(alpha=1.0)


def run_single_config(data, cv_strategy, config, seed):
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

        X_train_feat, X_test_feat = featurize(
            X_train, X_test, config["pst_variant"], config["kmer_args"]
        )

        model = make_model(config["model_type"])
        model.fit(X_train_feat, y_train.ravel())
        y_pred = model.predict(X_test_feat)

        metrics = regression_metrics(y_test.squeeze(), y_pred.squeeze())
        metrics["fold"] = k
        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"pst_kmer_shmushkovich_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir)

    configs = generate_configs()
    log("=" * 60)
    log("PST + KMersCounts Combined Features (Linear, paper settings)")
    log(f"Model types: standard (L2=0) + ridge (L2=1) — matching paper")
    log(f"Configs: 6 PST-only + 12 KMer-only + 36 combined = {len(configs)} total")
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
                r["pst_variant"] = str(config["pst_variant"])
                r["kmer_args"] = str(config["kmer_args"])
                r["model_type"] = config["model_type"]
                r["label"] = config["label"]
                all_rows.append(r)

            log(f"  Config {i:2d}: {config['label']:45s} seed={seed:3d} -> PCC={mean_pcc:.4f}")

        df = pd.DataFrame(all_rows)
        df.to_csv(run_dir / f"all_configs_{cv}.csv", index=False)

        grouped = df.groupby("config_id").agg(
            pcc_mean=("pearson_correlation", "mean"),
            pcc_std=("pearson_correlation", "std"),
            r2_mean=("r2_score", "mean"),
            rmse_mean=("root_mean_squared_error", "mean"),
            mae_mean=("mean_absolute_error", "mean"),
            spearman_mean=("spearman_correlation", "mean"),
            label=("label", "first"),
            pst_variant=("pst_variant", "first"),
            kmer_args=("kmer_args", "first"),
            model_type=("model_type", "first"),
            seed=("seed", "first"),
        ).sort_values("pcc_mean", ascending=False)

        grouped.to_csv(run_dir / f"ranked_configs_{cv}.csv")

        best = grouped.iloc[0]
        paper = PAPER_LINEAR[cv]

        log("")
        log(f"  BEST CONFIG ({cv} CV):")
        log(f"    Config:       {best['label']}")
        log(f"    Ours:         PCC = {best['pcc_mean']:.4f} +/- {best['pcc_std']:.4f}")
        log(f"    Paper Linear: PCC = {paper['pcc_mean']:.2f} +/- {paper['pcc_std']:.2f}")
        diff = best["pcc_mean"] - paper["pcc_mean"]
        log(f"    Delta:        {diff:+.4f}")

        log("")
        log(f"  TOP 10 CONFIGS ({cv} CV):")
        for rank, (_, row) in enumerate(grouped.head(10).iterrows()):
            log(f"    #{rank+1:2d}: {row['label']:45s} PCC={row['pcc_mean']:.4f} +/- {row['pcc_std']:.4f}")

        # Best from each category
        log("")
        for cat_name, mask_fn in [
            ("PST-only", lambda r: r["kmer_args"] == "None" and r["pst_variant"] != "None"),
            ("KMer-only", lambda r: r["pst_variant"] == "None" and r["kmer_args"] != "None"),
            ("PST+KMer", lambda r: r["pst_variant"] != "None" and r["kmer_args"] != "None"),
        ]:
            cat = grouped[grouped.apply(mask_fn, axis=1)]
            if len(cat) > 0:
                best_cat = cat.iloc[0]
                log(f"  Best {cat_name:10s}: {best_cat['label']:45s} PCC={best_cat['pcc_mean']:.4f} +/- {best_cat['pcc_std']:.4f}")

    log("")
    log(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
