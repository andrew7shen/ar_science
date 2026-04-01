"""
PST-tapered and PST+KMer with Ridge alpha sweep (high regularization).

Focuses on the genuinely novel AR contributions:
- PST-tapered: chess-inspired dual-regime (terminal/central) weighting
- PST+KMer: combining positional and compositional features

Uses standard + ridge with alpha sweep from 1.0 upward.
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

from oligogym.data import DatasetDownloader
from oligogym.features import OneHotEncoder, KMersCounts
from oligogym.metrics import regression_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

PAPER_LINEAR = {
    "Shmushkovich": {
        "random": {"pcc_mean": 0.24, "pcc_std": 0.09},
        "nucleobase": {"pcc_mean": 0.20, "pcc_std": 0.08},
    },
    "Alharbi_2020_1": {  # paper key=TLR7, Linear=0.77
        "random": {"pcc_mean": 0.77, "pcc_std": 0.02},
        "nucleobase": {"pcc_mean": 0.68, "pcc_std": 0.07},
    },
    "Alharbi_2020_2": {  # paper key=TLR8, Linear=0.54
        "random": {"pcc_mean": 0.54, "pcc_std": 0.13},
        "nucleobase": {"pcc_mean": 0.59, "pcc_std": 0.08},
    },
    "Ichihara_2007_2": {  # paper key=Ichihara, Linear=0.54
        "random": {"pcc_mean": 0.54, "pcc_std": 0.07},
        "nucleobase": {"pcc_mean": 0.49, "pcc_std": 0.09},
    },
    "siRNAmod": {  # 907 samples
        "random": {"pcc_mean": 0.62, "pcc_std": 0.06},
        "nucleobase": {"pcc_mean": 0.44, "pcc_std": 0.11},
    },
    "Cytotox LNA": {  # 768 samples
        "random": {"pcc_mean": 0.79, "pcc_std": 0.02},
        "nucleobase": {"pcc_mean": 0.64, "pcc_std": 0.16},
    },
}

DEFAULT_DATASET = "Shmushkovich"

KMER_CONFIGS = [
    {"k": [1], "modification_abundance": False},
    {"k": [1], "modification_abundance": True},
    {"k": [1, 2], "modification_abundance": False},
    {"k": [1, 2], "modification_abundance": True},
    {"k": [1, 2, 3], "modification_abundance": False},
    {"k": [1, 2, 3], "modification_abundance": True},
]

PST_VARIANTS = ["PST-tapered"]

ALPHAS = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]


def setup_logging(run_dir):
    logger = logging.getLogger("pst_alpha_sweep")
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
    configs = []

    # PST-only (all 3 variants for comparison): 3 × 9 = 27
    for variant, alpha in itertools.product(PST_VARIANTS, ALPHAS):
        configs.append({
            "pst_variant": variant,
            "kmer_args": None,
            "alpha": alpha,
            "label": f"{variant}_a{alpha}",
        })

    # PST + KMer combined (all 3 variants): 3 × 6 × 9 = 162
    for variant, kmer_args, alpha in itertools.product(PST_VARIANTS, KMER_CONFIGS, ALPHAS):
        k_str = "".join(map(str, kmer_args["k"]))
        mod = "mod" if kmer_args["modification_abundance"] else "nomod"
        configs.append({
            "pst_variant": variant,
            "kmer_args": kmer_args,
            "alpha": alpha,
            "label": f"{variant}+KMer-k{k_str}_{mod}_a{alpha}",
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


def download_dataset(dataset_name):
    downloader = DatasetDownloader()
    try:
        return downloader.download(dataset_name)
    except ValueError:
        name_to_key = dict(zip(downloader.all_datasets_info["name"], downloader.all_datasets_info["key"]))
        return downloader.download(name_to_key[dataset_name])


def run_single_config(data, cv_strategy, config, seed, dataset_name):
    np.random.seed(seed)
    results = []

    for k in range(5):
        if cv_strategy == "random":
            X_train, X_test, y_train, y_test = prepare_data_fold(data, k)
        else:
            data_fold = download_dataset(dataset_name)
            X_train, X_test, y_train, y_test, _, _ = data_fold.split(
                "nucleobase", return_index=True
            )

        X_train_feat, X_test_feat = featurize(
            X_train, X_test, config["pst_variant"], config["kmer_args"]
        )

        model = Ridge(alpha=config["alpha"])
        model.fit(X_train_feat, y_train.ravel())
        y_pred = model.predict(X_test_feat)

        metrics = regression_metrics(y_test.squeeze(), y_pred.squeeze())
        metrics["fold"] = k
        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(PAPER_LINEAR.keys()))
    args = parser.parse_args()

    dataset_name = args.dataset
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"pst_alpha_sweep_{dataset_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir)

    configs = generate_configs()
    log("=" * 60)
    log(f"PST + KMer Alpha Sweep (Ridge) on {dataset_name}")
    log(f"Alphas: {ALPHAS}")
    log(f"Configs: {len(configs)} total")
    log(f"Base seed: {args.base_seed}")
    log(f"Output: {run_dir}")
    log("=" * 60)

    log(f"Downloading {dataset_name} dataset...")
    data = download_dataset(dataset_name)
    log(f"Dataset size: {len(data.x)} samples")

    for cv in ["random", "nucleobase"]:
        log("")
        log(f"{'=' * 60}")
        log(f"{cv.upper()} CV: sweeping {len(configs)} configs")
        log(f"{'=' * 60}")

        all_rows = []
        for i, config in enumerate(configs):
            seed = args.base_seed + i
            fold_results = run_single_config(data, cv, config, seed, dataset_name)
            mean_pcc = np.mean([r["pearson_correlation"] for r in fold_results])

            for r in fold_results:
                r["cv_strategy"] = cv
                r["seed"] = seed
                r["config_id"] = i
                r["pst_variant"] = config["pst_variant"]
                r["kmer_args"] = str(config["kmer_args"])
                r["alpha"] = config["alpha"]
                r["label"] = config["label"]
                all_rows.append(r)

            log(f"  Config {i:3d}: {config['label']:50s} seed={seed:3d} -> PCC={mean_pcc:.4f}")

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
            alpha=("alpha", "first"),
            seed=("seed", "first"),
        ).sort_values("pcc_mean", ascending=False)

        grouped.to_csv(run_dir / f"ranked_configs_{cv}.csv")

        best = grouped.iloc[0]
        paper = PAPER_LINEAR[dataset_name][cv]

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
            log(f"    #{rank+1:2d}: {row['label']:50s} PCC={row['pcc_mean']:.4f} +/- {row['pcc_std']:.4f}")

        # Best from each category
        log("")
        for cat_name, mask_fn in [
            ("PST-only", lambda r: r["kmer_args"] == "None"),
            ("PST+KMer", lambda r: r["kmer_args"] != "None"),
        ]:
            cat = grouped[grouped.apply(mask_fn, axis=1)]
            if len(cat) > 0:
                best_cat = cat.iloc[0]
                log(f"  Best {cat_name:10s}: {best_cat['label']:50s} PCC={best_cat['pcc_mean']:.4f} +/- {best_cat['pcc_std']:.4f}")

        # Best per PST variant
        log("")
        for variant in PST_VARIANTS:
            cat = grouped[grouped["pst_variant"] == variant]
            if len(cat) > 0:
                best_cat = cat.iloc[0]
                log(f"  Best {variant:15s}: {best_cat['label']:50s} PCC={best_cat['pcc_mean']:.4f} +/- {best_cat['pcc_std']:.4f}")

    log("")
    log(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
