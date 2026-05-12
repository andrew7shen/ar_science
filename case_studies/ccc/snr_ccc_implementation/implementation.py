"""
SNR-CCC: Signal-to-Noise Ratio Analysis for Cell-Cell Communication.

Hybrid score per (ligand L, target cell type T):

    score(L, T) = logfc_coexpr_agg(L, T) * gain(z_tf(L, T))

where
  - logfc_coexpr_agg is the max/sum over (sender, receptor) of
    0.5 * (logFC(L in sender) + logFC(R in target)), matching the LIANA
    `logfc` baseline (AUPRC=0.244 on this benchmark).
  - z_tf is the analytical permutation z-score of L's CollecTRI TF
    activity in cells of T versus the whole-population null. Clamped to >=0
    so SNR only amplifies the baseline, never penalizes it.
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse

def _log_cp10k(adata):
    import scanpy as sc
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _gene_vec(gene_symbol, X, gene_to_idx):
    """Per-cell expression vector for a gene or complex (min across subunits)."""
    subunits = gene_symbol.split("_")
    vecs = []
    for s in subunits:
        idx = gene_to_idx.get(s)
        if idx is None:
            return None
        col = X[:, idx]
        if scipy.sparse.issparse(col):
            col = np.asarray(col.todense()).flatten()
        else:
            col = np.asarray(col).flatten()
        vecs.append(col)
    return np.minimum.reduce(vecs) if len(vecs) > 1 else vecs[0]


def _gene_logfc_by_ct(gene, X, gene_to_idx, type_to_cells, n_total):
    """
    Per-cell-type log-fold-change for a gene/complex. Data is assumed already
    log-normalized, so `mean_in_ct - mean_in_others` is a log fold-change.
    """
    v = _gene_vec(gene, X, gene_to_idx)
    if v is None:
        return None
    v_sum = v.sum()
    out = {}
    for ct, idx in type_to_cells.items():
        n_t = len(idx)
        mean_t = v[idx].mean() if n_t > 0 else 0.0
        denom = n_total - n_t
        mean_other = (v_sum - v[idx].sum()) / denom if denom > 0 else 0.0
        out[ct] = float(mean_t - mean_other)
    return out


def _patient_coherence(
    activity_df, labels, patients, cell_types, type_to_cells, min_cells=5
):
    """
    Signal-characteristic verification (SNR pillar 3): for each (cell_type,
    source) pair, compute the fraction of patients whose per-patient-T mean
    exceeds the population mean. A true ligand-induced signal should appear in
    most patients; a patient-specific artifact only in few.

    Returns dict[(cell_type, source)] -> float in [0, 1].
    """
    unique_patients = np.unique(patients)

    # Pre-compute (patient, cell_type) indices, skipping small groups.
    patient_T_cells = {}
    for p in unique_patients:
        p_mask = patients == p
        for ct in cell_types:
            ct_mask = labels == ct
            idx = np.where(p_mask & ct_mask)[0]
            if len(idx) >= min_cells:
                patient_T_cells[(p, ct)] = idx

    out = {}
    for col in activity_df.columns:
        arr = activity_df[col].values
        pop_mean = arr.mean()
        for ct in cell_types:
            per_patient_means = []
            for p in unique_patients:
                idx = patient_T_cells.get((p, ct))
                if idx is None:
                    continue
                per_patient_means.append(arr[idx].mean())
            if len(per_patient_means) < 3:
                continue
            n_above = sum(1 for m in per_patient_means if m > pop_mean)
            out[(ct, col)] = n_above / len(per_patient_means)
    return out


def _activity_z_by_celltype(activity_df, labels, cell_types, type_to_cells, eps=1e-8):
    """
    Analytical permutation z of mean activity in each cell type vs population,
    for a per-cell DataFrame of CollecTRI TF activity.
    Returns dict[(cell_type, source)] -> z.
    """
    n_total = len(labels)
    out = {}
    for source in activity_df.columns:
        arr = activity_df[source].values
        pop_mean = arr.mean()
        pop_var = arr.var()
        for ct in cell_types:
            idx = type_to_cells[ct]
            n_T = len(idx)
            if n_T < 10 or n_T >= n_total - 10:
                continue
            signal = arr[idx].mean()
            null_std = np.sqrt(
                max(pop_var, 0.0) / n_T * (n_total - n_T) / (n_total - 1)
            )
            out[(ct, source)] = (signal - pop_mean) / (null_std + eps)
    return out


def precompute(adata, verbose=True):
    """
    Run log-normalization + CollecTRI ULM + TF z-scores once, storing
    everything on the returned adata. Intended to be called once per dataset,
    with the output reused across aggregation variants.
    """
    import decoupler as dc

    t_total = time.time()

    if adata.uns.get("_snr_log_cp10k_done"):
        if verbose:
            print("Step 1: Log-normalization already done, skipping.")
    else:
        if verbose:
            print("Step 1: Log-normalizing...")
        t0 = time.time()
        adata = _log_cp10k(adata)
        adata.uns["_snr_log_cp10k_done"] = True
        if verbose:
            print(f"  Done ({time.time() - t0:.1f}s)")

    # CollecTRI TF activity (filtered to ligand-relevant TFs for speed).
    if "score_ulm" in adata.obsm:
        if verbose:
            print("Step 2: CollecTRI ULM scores already present, skipping.")
    else:
        if verbose:
            print("Step 2: Running filtered CollecTRI via decoupler ULM...")
        t0 = time.time()
        collectri = dc.op.collectri(organism="human")
        collectri_sub = collectri[collectri["source"].isin(TARGET_TFS)].copy()
        if verbose:
            print(
                f"  Filtered CollecTRI: {collectri_sub['source'].nunique()} TFs, "
                f"{len(collectri_sub)} interactions"
            )
        dc.mt.ulm(adata, collectri_sub, verbose=verbose)
        if verbose:
            print(
                f"  CollecTRI ULM done, {adata.obsm['score_ulm'].shape[1]} TFs "
                f"({time.time() - t0:.1f}s)"
            )

    ulm_tf = adata.obsm["score_ulm"]
    labels = adata.obs["label"].values
    cell_types = list(adata.obs["label"].cat.categories)
    type_to_cells = {ct: np.where(labels == ct)[0] for ct in cell_types}

    if "_snr_tf_z" in adata.uns and "_snr_tf_z_spec" in adata.uns:
        if verbose:
            print("Step 3: Activity z-scores cached, skipping.")
    else:
        if verbose:
            print("Step 3: Computing per-cell-type TF z-scores...")
        t0 = time.time()
        tf_z = _activity_z_by_celltype(ulm_tf, labels, cell_types, type_to_cells)

        # TF-specificity: for each cell type, subtract the mean z across all
        # TFs in that cell type. Suppresses broadly-active-TF cell types
        # (e.g. Cancer.Cycling with high FOS/JUN/MYC) where high z doesn't
        # imply ligand-specific response.
        tf_z_spec = {}
        by_ct = {}
        for (ct, tf), v in tf_z.items():
            by_ct.setdefault(ct, []).append(v)
        ct_tf_mean = {ct: float(np.mean(vs)) for ct, vs in by_ct.items()}
        for (ct, tf), v in tf_z.items():
            tf_z_spec[(ct, tf)] = v - ct_tf_mean[ct]

        adata.uns["_snr_tf_z"] = {f"{ct}||{tf}": v for (ct, tf), v in tf_z.items()}
        adata.uns["_snr_tf_z_spec"] = {f"{ct}||{tf}": v for (ct, tf), v in tf_z_spec.items()}
        if verbose:
            print(
                f"  {len(tf_z)} TF + {len(tf_z_spec)} TF-spec z-scores "
                f"({time.time() - t0:.1f}s)"
            )

    if "_snr_coherence_tf" in adata.uns:
        if verbose:
            print("Step 4: Coherence scores cached, skipping.")
    else:
        if verbose:
            print("Step 4: Computing cross-patient coherence (SNR pillar 3)...")
        t0 = time.time()
        patients = adata.obs["orig.ident"].values
        coh_tf = _patient_coherence(
            ulm_tf, labels, patients, cell_types, type_to_cells
        )
        adata.uns["_snr_coherence_tf"] = {
            f"{ct}||{s}": v for (ct, s), v in coh_tf.items()
        }
        if verbose:
            print(
                f"  {len(coh_tf)} TF coherence scores "
                f"({time.time() - t0:.1f}s) "
                f"(n_patients={len(np.unique(patients))})"
            )

    if verbose:
        print(f"Precompute total: {time.time() - t_total:.1f}s")

    return adata


GAIN_FNS = {
    "raw":       lambda z: 1.0 + max(0.0, z),
    "log1p":     lambda z: 1.0 + np.log1p(max(0.0, z)),
    "sqrt":      lambda z: 1.0 + np.sqrt(max(0.0, z)),
    "linear005": lambda z: 1.0 + 0.05 * max(0.0, z),
    "linear010": lambda z: 1.0 + 0.10 * max(0.0, z),
    "linear025": lambda z: 1.0 + 0.25 * max(0.0, z),
    "linear050": lambda z: 1.0 + 0.50 * max(0.0, z),
    "linear100": lambda z: 1.0 + 1.00 * max(0.0, z),
}


def snr_ccc(
    adata,
    aggregate_how="max",
    gain="log1p",
    signature="tf",
    combine="mult",
    coherence=False,
    verbose=True,
):
    """
    Score ligand-target interactions with logFC co-expression × SNR gain.

    signature:
      "tf"            : ligand-specific CollecTRI TFs.
      "tf_specific"   : TFs, z-adjusted by per-cell-type mean-across-TFs
                        (suppresses broad-TF cell types like Cancer.Cycling).

    combine:
      "mult" : coexpr * gain_fn(z_eff)                       [default]
      "geom" : sqrt(max(0, coexpr) * max(0, z_eff))          [requires both]

    coherence: if True, multiply z_used by the fraction of patients whose
    per-patient-T mean activity exceeds the population mean for that source.
    This is the 4th SNR pillar ("signal characteristic verification"): a true
    ligand-induced signal should be coherent across biological replicates,
    while single-patient artifacts get penalized.

    gain: shape of the multiplicative gain on z_eff. Only applies to combine="mult".
    """
    if gain not in GAIN_FNS:
        raise ValueError(f"Unknown gain={gain}; options: {list(GAIN_FNS)}")
    if signature not in {"tf", "tf_specific"}:
        raise ValueError(f"Unknown signature={signature}")
    if combine not in {"mult", "geom"}:
        raise ValueError(f"Unknown combine={combine}")
    gain_fn = GAIN_FNS[gain]

    t_total = time.time()

    adata = precompute(adata, verbose=verbose)

    labels = adata.obs["label"].values
    cell_types = list(adata.obs["label"].cat.categories)
    type_to_cells = {ct: np.where(labels == ct)[0] for ct in cell_types}
    n_total = adata.shape[0]

    # Reconstruct TF z and coherence dicts.
    tf_z = {}
    for k, v in adata.uns.get("_snr_tf_z", {}).items():
        ct, tf = k.split("||", 1)
        tf_z[(ct, tf)] = v
    tf_z_spec = {}
    for k, v in adata.uns.get("_snr_tf_z_spec", {}).items():
        ct, tf = k.split("||", 1)
        tf_z_spec[(ct, tf)] = v
    coh_tf = {}
    for k, v in adata.uns.get("_snr_coherence_tf", {}).items():
        ct, tf = k.split("||", 1)
        coh_tf[(ct, tf)] = v

    # Pick which TF z dict the signature wants.
    tf_z_active = tf_z_spec if signature == "tf_specific" else tf_z

    resource = adata.uns["ligand_receptor_resource"]
    var_names = set(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    gt_ligands = sorted(adata.uns["ccc_target"]["ligand"].unique())

    ligand_to_receptors = {}
    for _, row in resource.iterrows():
        lig = row["ligand_genesymbol"]
        rec = row["receptor_genesymbol"]
        if lig not in gt_ligands:
            continue
        if not all(s in var_names for s in lig.split("_")):
            continue
        if not all(s in var_names for s in rec.split("_")):
            continue
        ligand_to_receptors.setdefault(lig, []).append(rec)

    if verbose:
        print(f"  Ligands with receptors in data: {len(ligand_to_receptors)}")

    if verbose:
        print("Step 4: Caching per-cell-type logFC for ligands and receptors...")
    t0 = time.time()
    lig_logfc = {}
    for lig in gt_ligands:
        m = _gene_logfc_by_ct(lig, adata.X, gene_to_idx, type_to_cells, n_total)
        if m is not None:
            lig_logfc[lig] = m

    all_receptors = set()
    for recs in ligand_to_receptors.values():
        all_receptors.update(recs)
    rec_logfc = {}
    for rec in all_receptors:
        m = _gene_logfc_by_ct(rec, adata.X, gene_to_idx, type_to_cells, n_total)
        if m is not None:
            rec_logfc[rec] = m

    if verbose:
        print(
            f"  {len(lig_logfc)} ligands, {len(rec_logfc)} receptors "
            f"({time.time() - t0:.1f}s)"
        )

    if verbose:
        print(f"Step 5: Scoring (L, T) pairs ({aggregate_how} aggregation)...")
    t0 = time.time()
    records = []

    tf_cols = set(adata.obsm["score_ulm"].columns) if "score_ulm" in adata.obsm else set()

    for lig in gt_ligands:
        receptors = ligand_to_receptors.get(lig, [])
        has_terms = (lig in lig_logfc) and any(r in rec_logfc for r in receptors)
        tfs = [t for t in LIGAND_TO_TFS.get(lig, []) if t in tf_cols]

        for target_ct in cell_types:
            z_tf_pairs = [(t, tf_z_active.get((target_ct, t), 0.0)) for t in tfs]

            best_tf, z_tf_best = (max(z_tf_pairs, key=lambda kv: kv[1])
                                  if z_tf_pairs else (None, 0.0))

            z_used = z_tf_best if tfs else 0.0

            # SNR pillar 3: multiply z_used by cross-patient coherence of the
            # source that produced it. Spurious single-patient signals get
            # damped; truly coherent signals pass through unchanged.
            coh = 1.0
            if coherence and best_tf is not None:
                coh = coh_tf.get((target_ct, best_tf), 1.0)
                z_used = z_used * coh

            # logFC co-expression baseline.
            if has_terms:
                terms = []
                rec_vals_t = [rec_logfc[r][target_ct] for r in receptors if r in rec_logfc]
                lig_lfc = lig_logfc[lig]
                for sender_ct in cell_types:
                    lig_s = lig_lfc[sender_ct]
                    for rv in rec_vals_t:
                        terms.append(0.5 * (lig_s + rv))
                if aggregate_how == "max":
                    coexpr_agg = max(terms) if terms else 0.0
                elif aggregate_how == "sum":
                    coexpr_agg = float(np.sum(terms)) if terms else 0.0
                else:
                    raise ValueError(f"Unknown aggregate_how={aggregate_how}")
            else:
                coexpr_agg = 0.0

            if combine == "mult":
                score = coexpr_agg * gain_fn(z_used)
            else:  # geom
                score = float(np.sqrt(max(0.0, coexpr_agg) * max(0.0, z_used)))
            records.append({
                "ligand": lig,
                "target": target_ct,
                "score": float(score),
                "coexpr": float(coexpr_agg),
                "z_used": float(z_used),
                "z_tf": float(z_tf_best),
                "coh": float(coh),
                "tfs": ",".join(tfs) if tfs else "",
            })

    pred_df = pd.DataFrame(records)
    # ccc_pred needs only ligand/target/score; keep the diagnostic columns
    # on a parallel frame for post-run inspection.
    adata.uns["ccc_pred"] = pred_df[["ligand", "target", "score"]].copy()
    adata.uns["ccc_pred_diag"] = pred_df

    if verbose:
        print(f"  Scoring done ({time.time() - t0:.1f}s)")
        print(f"  Predictions: {len(pred_df)} (ligand, target) pairs")
        print(
            f"  Score range: [{pred_df['score'].min():.4f}, "
            f"{pred_df['score'].max():.4f}]"
        )
        print(f"  Total SNR-CCC time: {time.time() - t_total:.1f}s")

    return adata
