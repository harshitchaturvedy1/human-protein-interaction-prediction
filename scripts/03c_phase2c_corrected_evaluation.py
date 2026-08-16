"""
Phase 2c: Corrected Protein-Disjoint Evaluation
============================================================================
Fixes two bugs found in Phase 2b during review:

BUG 1 (Phase 2b): split_group = min(id_a, id_b) does NOT guarantee a
protein-disjoint split. A protein can be the "min" in one pair (forcing
that pair into one split) and the "max" in another pair whose min sends
it to the other split -- so the same protein leaks across train/test.
Reviewer found 354 overlapping proteins on a reproduction run.

FIX: Partition the protein SET first (not the pairs). Assign every unique
protein to train or test via GroupShuffleSplit over protein IDs, then keep
only pairs where BOTH proteins fall in the same partition. Any pair that
straddles the partition boundary is dropped, not assigned. Overlap is
asserted to be exactly zero and saved to the results file.

BUG 2 (Phase 2b): known_pairs was built only from the 968 sampled positives,
not the full BioGRID release. Reviewer found 46 of 968 sampled "negatives"
are actually known human-human physical interactions elsewhere in BioGRID.

FIX: Build canonical (min_id, max_id) pairs from the FULL filtered BioGRID
file (biogrid_filtered.csv), not just the sample. Drop self-pairs and
duplicates. Sample negatives as unique pairs not in this canonical set,
without replacement. These are labeled "unlabeled/non-observed pairs" in
all output, NOT "confirmed negatives" -- absence from BioGRID is not proof
of non-interaction.

ADDITIONAL: Runs across multiple seeds and reports mean/std for ROC-AUC,
AUPRC, test prevalence, precision@K, and both control baselines, so a
single lucky/unlucky split isn't mistaken for a stable result.

============================================================================
ROUND 2 FIXES (post-review of the first 03c version):

1. Positive pairs are now canonicalized to (min_id, max_id) and
   deduplicated BEFORE splitting, so reversed A-B/B-A duplicates can't
   land on opposite sides of the partition or inflate pair counts.

2. The protein-overlap assertion now runs on the FINAL train_all/test_all
   data (after negative sampling and row filtering), not on the
   pre-split train_proteins/test_proteins sets. The old check was
   tautological -- those two sets are disjoint by construction regardless
   of what happens downstream, so it could never have caught a real leak.

3. sample_negatives() calls are now asserted to return exactly the number
   of pairs requested, with no duplicates. Previously a pool that hit
   max_attempts could silently return fewer negatives than positives,
   producing a class-imbalanced set without warning.

4. Bootstrap 95% CIs (2000 resamples) are reported per seed for ROC-AUC,
   given the small test-set size (~80 rows) makes single point estimates
   and even the 5-seed std potentially misleading.

5. Per-seed positive/unlabeled counts are reported explicitly.

6. A note is added clarifying that AUPRC here reflects a 1:1 constructed
   test set, not real biological screening prevalence.
============================================================================

ROUND 3 ADDITIONS (final checklist before repo freeze):

- Sanity checks that fail loudly (assert) if: a protein appears in both
  final train/test data; a sampled negative is actually a known BioGRID
  pair; a duplicate/reversed pair exists within a split; swapping Protein
  A/B changes the feature vector by more than floating-point tolerance;
  either split ends up with only one class.
- Additional saved outputs: phase2c_metrics_by_seed.csv (structured,
  one row per seed/model), feature_importance_fixed.csv (per-seed +
  mean importance), phase2c_test_predictions.csv (per-row predictions
  and labels, for independent CI recomputation), phase2c_split_assignments.csv
  (train/test protein-pair assignments per seed, IDs only -- no sequences,
  so file size stays small).
============================================================================

Inputs:
  biogrid_sample_1000_with_sequences.csv  (positive pairs + sequences)
  biogrid_filtered.csv                    (full filtered release, for
                                            canonical known-pairs -- large
                                            file, only Interactor ID columns
                                            are read)
Outputs:
  phase2c_results.txt
  feature_importance_fixed.csv  (from seed 42 run, for inspection)

Install: pip install pandas numpy scikit-learn xgboost biopython
Usage:   python 03c_phase2c_corrected_evaluation.py
"""

import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

SEEDS = [42, 7, 123, 2024, 99]

# ============================================================================
# STEP 1: Load positive pairs (sample) + sequences
# ============================================================================
print("Step 1: Loading positive pairs with sequences...")
df = pd.read_csv("biogrid_sample_1000_with_sequences.csv")

ID_A_COL, ID_B_COL = "Official Symbol Interactor A", "Official Symbol Interactor B"
SEQ_A_COL, SEQ_B_COL = "Sequence Interactor A", "Sequence Interactor B"

df = df.rename(columns={
    ID_A_COL: "id_a", ID_B_COL: "id_b",
    SEQ_A_COL: "seq_a", SEQ_B_COL: "seq_b"
})
df = df.dropna(subset=["id_a", "id_b", "seq_a", "seq_b"]).reset_index(drop=True)
# Drop self-pairs (a protein "interacting" with itself is not a useful PPI example here)
df = df[df["id_a"] != df["id_b"]].reset_index(drop=True)
n_before_canon = len(df)

# Canonicalize: reorder every pair as (min_id, max_id) so that A-B and B-A
# rows collapse to the same key, then drop duplicates. Without this, a
# reversed duplicate could land on opposite sides of the protein-disjoint
# split as two "different" pairs while actually sharing both proteins, or
# inflate counts if both directions were present in the raw sample.
def canonicalize_row(row):
    if row["id_a"] <= row["id_b"]:
        return row["id_a"], row["id_b"], row["seq_a"], row["seq_b"]
    else:
        return row["id_b"], row["id_a"], row["seq_b"], row["seq_a"]

canon = df.apply(canonicalize_row, axis=1, result_type="expand")
canon.columns = ["id_a", "id_b", "seq_a", "seq_b"]
df = canon.drop_duplicates(subset=["id_a", "id_b"]).reset_index(drop=True)
n_dupes_dropped = n_before_canon - len(df)
print(f"  Loaded {n_before_canon} positive pairs (after dropping self-pairs); "
      f"{n_dupes_dropped} duplicate/reversed pairs removed -> {len(df)} unique canonical positives")

# ============================================================================
# STEP 2: Build canonical known-pairs set from the FULL filtered BioGRID
# release, not just the 1000-pair sample. This is what negatives get
# checked against, so a sampled "negative" that's actually a known
# interaction anywhere in BioGRID gets excluded.
# ============================================================================
print("\nStep 2: Building canonical known-pairs set from full BioGRID release...")
# Only read the two ID columns -- the full file is large and we don't need
# anything else from it for this step.
full_ids = pd.read_csv(
    "biogrid_filtered.csv",
    usecols=[ID_A_COL, ID_B_COL],
    low_memory=False
).rename(columns={ID_A_COL: "id_a", ID_B_COL: "id_b"}).dropna()

full_ids = full_ids[full_ids["id_a"] != full_ids["id_b"]]

canonical_pairs = set(
    zip(
        np.minimum(full_ids["id_a"], full_ids["id_b"]),
        np.maximum(full_ids["id_a"], full_ids["id_b"])
    )
)
print(f"  {len(full_ids)} total pairs in full release -> "
      f"{len(canonical_pairs)} unique canonical (min,max) pairs")

# Sanity check against the bug the reviewer found: how many of our sampled
# "negatives" from the OLD (Phase 2b) approach would have been contaminated?
# (Informational only -- Phase 2c does not reuse Phase 2b's negative set.)

# ============================================================================
# STEP 3: Protein-set partition (the actual disjoint-split fix)
# Partition unique PROTEINS into train/test first. A pair is only kept if
# BOTH its proteins landed in the same partition; pairs straddling the
# boundary are dropped entirely rather than assigned to either side.
# ============================================================================
def build_protein_disjoint_split(pairs_df, seed, test_size=0.2):
    all_proteins = pd.unique(pd.concat([pairs_df["id_a"], pairs_df["id_b"]]))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_proteins)
    n_test = int(len(shuffled) * test_size)
    test_proteins = set(shuffled[:n_test])
    train_proteins = set(shuffled[n_test:])

    def side(pid):
        if pid in train_proteins:
            return "train"
        if pid in test_proteins:
            return "test"
        return None

    train_rows, test_rows, dropped = [], [], 0
    for _, row in pairs_df.iterrows():
        sa, sb = side(row["id_a"]), side(row["id_b"])
        if sa == sb == "train":
            train_rows.append(row)
        elif sa == sb == "test":
            test_rows.append(row)
        else:
            dropped += 1  # straddles the boundary -- excluded, not assigned

    train_df = pd.DataFrame(train_rows).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).reset_index(drop=True)
    return train_df, test_df, train_proteins, test_proteins, dropped


# ============================================================================
# STEP 4: Sample unlabeled/non-observed negative pairs WITHIN a given
# protein set (so negatives for train only use train-side proteins, and
# negatives for test only use test-side proteins -- keeps the split
# disjoint end-to-end instead of leaking through the negatives).
# ============================================================================
def sample_negatives(protein_pool_df, n_needed, seed):
    """protein_pool_df: columns [id, seq], proteins confined to one split side."""
    rng = np.random.RandomState(seed)
    proteins = protein_pool_df.reset_index(drop=True)
    n = len(proteins)
    seen_local = set()
    neg_rows = []
    attempts = 0
    max_attempts = n_needed * 200
    while len(neg_rows) < n_needed and attempts < max_attempts and n >= 2:
        attempts += 1
        i, j = rng.choice(n, size=2, replace=False)
        p1, p2 = proteins.iloc[i], proteins.iloc[j]
        key = (min(p1["id"], p2["id"]), max(p1["id"], p2["id"]))
        if key in canonical_pairs or key in seen_local:
            continue
        seen_local.add(key)
        neg_rows.append({"id_a": p1["id"], "id_b": p2["id"], "seq_a": p1["seq"], "seq_b": p2["seq"]})
    return pd.DataFrame(neg_rows)


# ============================================================================
# STEP 5: Symmetric feature engineering (unchanged from Phase 2b -- this
# part was not flagged as broken)
# ============================================================================
def seq_props(seq):
    try:
        pa = ProteinAnalysis(str(seq).replace("*", "").replace("X", ""))
        return {
            "len": len(seq), "mw": pa.molecular_weight(), "arom": pa.aromaticity(),
            "inst": pa.instability_index(), "gravy": pa.gravy(), "pi": pa.isoelectric_point(),
        }
    except Exception:
        return {k: np.nan for k in ["len", "mw", "arom", "inst", "gravy", "pi"]}

PROP_NAMES = ["len", "mw", "arom", "inst", "gravy", "pi"]
FEATURE_COLS = [f"{p}_sum" for p in PROP_NAMES] + [f"{p}_absdiff" for p in PROP_NAMES]

def add_symmetric_features(pairs_df):
    props_a = pairs_df["seq_a"].apply(seq_props).apply(pd.Series).add_suffix("_a")
    props_b = pairs_df["seq_b"].apply(seq_props).apply(pd.Series).add_suffix("_b")
    out = pairs_df.copy()
    for prop in PROP_NAMES:
        out[f"{prop}_sum"] = props_a[f"{prop}_a"] + props_b[f"{prop}_b"]
        out[f"{prop}_absdiff"] = (props_a[f"{prop}_a"] - props_b[f"{prop}_b"]).abs()
    out = out.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Sanity check: swapping Protein A and Protein B must not change the
    # feature vector. Verify on a small random sample rather than every row,
    # to keep this cheap.
    if len(out) > 0:
        check_n = min(15, len(out))
        check_rows = out.sample(check_n, random_state=1)
        swapped = check_rows.rename(columns={"id_a": "id_b", "id_b": "id_a", "seq_a": "seq_b", "seq_b": "seq_a"})
        swapped = swapped[["id_a", "id_b", "seq_a", "seq_b"]]
        # Recompute features directly on the swapped rows (avoids recursion)
        sw_props_a = swapped["seq_a"].apply(seq_props).apply(pd.Series).add_suffix("_a")
        sw_props_b = swapped["seq_b"].apply(seq_props).apply(pd.Series).add_suffix("_b")
        max_drift = 0.0
        for prop in PROP_NAMES:
            orig_sum = check_rows[f"{prop}_sum"].values
            swap_sum = (sw_props_a[f"{prop}_a"] + sw_props_b[f"{prop}_b"]).values
            orig_diff = check_rows[f"{prop}_absdiff"].values
            swap_diff = (sw_props_a[f"{prop}_a"] - sw_props_b[f"{prop}_b"]).abs().values
            max_drift = max(max_drift, np.abs(orig_sum - swap_sum).max(), np.abs(orig_diff - swap_diff).max())
        assert max_drift < 1e-6, (
            f"Swap-invariance check failed: max feature drift {max_drift:.2e} after "
            f"swapping Protein A/B -- symmetric feature construction is broken"
        )
    return out


def precision_at_k(y_true, y_scores, k):
    order = np.argsort(-y_scores)[:k]
    return y_true.values[order].mean()


def bootstrap_auc_ci(y_true, y_scores, n_boot=2000, ci=0.95, seed=0):
    """Bootstrap resample test rows (with replacement) and recompute ROC-AUC
    each time, to get a 95% CI that reflects the small test-set size (~80
    rows) rather than presenting a single point estimate as if it were
    precise."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resamples with only one class present
        boot_aucs.append(roc_auc_score(yt, ys))
    lo = np.percentile(boot_aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_aucs, (1 + ci) / 2 * 100)
    return lo, hi, len(boot_aucs)


# ============================================================================
# STEP 6: Run the full pipeline once per seed
# ============================================================================
def run_seed(seed):
    train_pos, test_pos, train_proteins, test_proteins, dropped = build_protein_disjoint_split(
        df[["id_a", "id_b", "seq_a", "seq_b"]], seed
    )
    # NOTE: this pre-check is tautological -- train_proteins/test_proteins are
    # the two halves of the SAME partition, so their intersection is 0 by
    # construction regardless of anything that happens downstream (negative
    # sampling, feature engineering, row drops). It's kept only as a cheap
    # early sanity check; the assertion that actually matters is below,
    # computed on the final constructed train_all/test_all.
    precheck_overlap = train_proteins & test_proteins
    assert len(precheck_overlap) == 0, f"Seed {seed}: partition itself is broken ({len(precheck_overlap)})"

    all_proteins_df = pd.concat([
        df[["id_a", "seq_a"]].rename(columns={"id_a": "id", "seq_a": "seq"}),
        df[["id_b", "seq_b"]].rename(columns={"id_b": "id", "seq_b": "seq"})
    ]).drop_duplicates(subset="id").reset_index(drop=True)

    train_pool = all_proteins_df[all_proteins_df["id"].isin(train_proteins)]
    test_pool = all_proteins_df[all_proteins_df["id"].isin(test_proteins)]

    train_neg = sample_negatives(train_pool, len(train_pos), seed)
    test_neg = sample_negatives(test_pool, len(test_pos), seed + 10000)

    # Fix 3: sample_negatives can silently return fewer than requested if it
    # hits max_attempts. Fail loudly instead of quietly training on an
    # under-sampled / imbalanced set.
    assert len(train_neg) == len(train_pos), (
        f"Seed {seed}: train negative sampling returned {len(train_neg)} pairs, "
        f"needed {len(train_pos)} -- increase max_attempts or shrink the pool constraint"
    )
    assert len(test_neg) == len(test_pos), (
        f"Seed {seed}: test negative sampling returned {len(test_neg)} pairs, "
        f"needed {len(test_pos)} -- increase max_attempts or shrink the pool constraint"
    )
    assert train_neg[["id_a", "id_b"]].apply(tuple, axis=1).nunique() == len(train_neg), \
        f"Seed {seed}: train negatives contain duplicate pairs"
    assert test_neg[["id_a", "id_b"]].apply(tuple, axis=1).nunique() == len(test_neg), \
        f"Seed {seed}: test negatives contain duplicate pairs"

    # Sanity check: no sampled negative is actually a known pair (canonical or
    # reversed) -- sample_negatives already filters against canonical_pairs,
    # this re-verifies it independently on the output.
    for neg_df, split_name in [(train_neg, "train"), (test_neg, "test")]:
        neg_keys = set(zip(
            np.minimum(neg_df["id_a"], neg_df["id_b"]),
            np.maximum(neg_df["id_a"], neg_df["id_b"])
        ))
        contaminated = neg_keys & canonical_pairs
        assert len(contaminated) == 0, (
            f"Seed {seed}: {len(contaminated)} {split_name} negative pairs are actually "
            f"known BioGRID interactions -- negative sampling is contaminated"
        )

    train_pos = train_pos.copy(); train_pos["label"] = 1
    test_pos = test_pos.copy(); test_pos["label"] = 1
    train_neg["label"] = 0
    test_neg["label"] = 0

    train_all = pd.concat([train_pos, train_neg], ignore_index=True)
    test_all = pd.concat([test_pos, test_neg], ignore_index=True)

    # Sanity check: no duplicate or reversed pair within the combined set
    # (positives + negatives together), in either split.
    for combined, split_name in [(train_all, "train"), (test_all, "test")]:
        keys = list(zip(
            np.minimum(combined["id_a"], combined["id_b"]),
            np.maximum(combined["id_a"], combined["id_b"])
        ))
        assert len(keys) == len(set(keys)), (
            f"Seed {seed}: {split_name} set contains duplicate/reversed pairs across pos+neg combined"
        )

    # Sanity check: each split must contain both classes (a single-class
    # split would make ROC-AUC undefined/meaningless).
    assert train_all["label"].nunique() == 2, f"Seed {seed}: train split has only one class"
    assert test_all["label"].nunique() == 2, f"Seed {seed}: test split has only one class"

    # The assertion that actually matters: compute protein overlap on the
    # FINAL data going into the models, not the pre-negative-sampling sets.
    final_train_proteins = set(train_all["id_a"]) | set(train_all["id_b"])
    final_test_proteins = set(test_all["id_a"]) | set(test_all["id_b"])
    final_overlap = final_train_proteins & final_test_proteins
    assert len(final_overlap) == 0, (
        f"Seed {seed}: protein overlap detected in FINAL train/test data "
        f"({len(final_overlap)} proteins) -- split is not actually disjoint"
    )

    train_feat = add_symmetric_features(train_all)
    test_feat = add_symmetric_features(test_all)

    X_train, y_train = train_feat[FEATURE_COLS], train_feat["label"]
    X_test, y_test = test_feat[FEATURE_COLS], test_feat["label"]

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, random_state=seed, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    len_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=seed, eval_metric="logloss")
    len_model.fit(X_train[["len_sum", "len_absdiff"]], y_train)
    len_auc = roc_auc_score(y_test, len_model.predict_proba(X_test[["len_sum", "len_absdiff"]])[:, 1])

    y_train_shuffled = y_train.sample(frac=1, random_state=seed + 5000).reset_index(drop=True)
    shuf_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=seed, eval_metric="logloss")
    shuf_model.fit(X_train, y_train_shuffled.values)
    shuf_auc = roc_auc_score(y_test, shuf_model.predict_proba(X_test)[:, 1])

    def metrics_for(model):
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        return {
            "roc_auc": roc_auc_score(y_test, proba),
            "auprc": average_precision_score(y_test, proba),
            "p_at_20": precision_at_k(y_test, proba, min(20, len(y_test))),
            "p_at_50": precision_at_k(y_test, proba, min(50, len(y_test))),
            "f1": f1_score(y_test, pred, average="weighted"),
            "accuracy": accuracy_score(y_test, pred),
            "cm": cm,
        }

    result = {
        "seed": seed,
        "n_train": len(train_feat), "n_test": len(test_feat),
        "n_train_pos": int((train_feat["label"] == 1).sum()),
        "n_train_neg": int((train_feat["label"] == 0).sum()),
        "n_test_pos": int((test_feat["label"] == 1).sum()),
        "n_test_neg": int((test_feat["label"] == 0).sum()),
        "test_prevalence": y_test.mean(),
        "dropped_pairs": dropped,
        "protein_overlap": len(final_overlap),
        "xgb": metrics_for(xgb_model),
        "rf": metrics_for(rf_model),
        "len_auc": len_auc,
        "shuf_auc": shuf_auc,
        "y_test": y_test.values,
        "xgb_proba": xgb_model.predict_proba(X_test)[:, 1],
        "rf_proba": rf_model.predict_proba(X_test)[:, 1],
        "test_id_a": test_feat["id_a"].values,
        "test_id_b": test_feat["id_b"].values,
        "train_id_a": train_feat["id_a"].values,
        "train_id_b": train_feat["id_b"].values,
        "importances": pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False),
    }
    return result


# ============================================================================
# STEP 7: Run all seeds, aggregate, write results
# ============================================================================
print(f"\nStep 3-6: Running {len(SEEDS)} seeds ({SEEDS})...")
all_results = []
for s in SEEDS:
    print(f"\n--- Seed {s} ---")
    r = run_seed(s)
    print(f"  Train: {r['n_train']} rows ({r['n_train_pos']} pos / {r['n_train_neg']} unlabeled), "
          f"Test: {r['n_test']} rows ({r['n_test_pos']} pos / {r['n_test_neg']} unlabeled), "
          f"dropped (boundary) pairs: {r['dropped_pairs']}")
    print(f"  Protein overlap in final train/test data (must be 0): {r['protein_overlap']}")
    print(f"  Test prevalence: {r['test_prevalence']:.3f}")
    xgb_lo, xgb_hi, xgb_nb = bootstrap_auc_ci(r["y_test"], r["xgb_proba"], seed=s)
    rf_lo, rf_hi, rf_nb = bootstrap_auc_ci(r["y_test"], r["rf_proba"], seed=s)
    r["xgb_ci"] = (xgb_lo, xgb_hi)
    r["rf_ci"] = (rf_lo, rf_hi)
    print(f"  XGBoost ROC-AUC: {r['xgb']['roc_auc']:.4f}  [95% CI {xgb_lo:.4f}-{xgb_hi:.4f}]  AUPRC: {r['xgb']['auprc']:.4f}")
    print(f"  RF ROC-AUC:      {r['rf']['roc_auc']:.4f}  [95% CI {rf_lo:.4f}-{rf_hi:.4f}]  AUPRC: {r['rf']['auprc']:.4f}")
    print(f"  Length-only AUC: {r['len_auc']:.4f}  Shuffled-label AUC: {r['shuf_auc']:.4f}")
    all_results.append(r)

def agg(key_path):
    vals = []
    for r in all_results:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    return np.mean(vals), np.std(vals)

lines = []
lines.append("PHASE 2C: CORRECTED PROTEIN-DISJOINT EVALUATION")
lines.append("=" * 70)
lines.append(f"Seeds: {SEEDS}")
lines.append("")
lines.append("NOTE: Negative pairs are UNLABELED/NON-OBSERVED pairs (random protein")
lines.append("pairs not found in the full filtered BioGRID release), not confirmed")
lines.append("negatives. Absence from BioGRID is not proof of non-interaction.")
lines.append("")
lines.append(f"Protein overlap across all seeds, checked on final train/test data (must all be 0): "
             f"{[r['protein_overlap'] for r in all_results]}")
lines.append("")

lines.append("NOTE ON AUPRC: the test set is constructed 1:1 positive:unlabeled by design,")
lines.append("so the no-skill AUPRC baseline here is ~0.5. This AUPRC is only meaningful")
lines.append("for this constructed, balanced candidate set -- it does NOT estimate")
lines.append("precision under the much lower positive prevalence expected in a real")
lines.append("biological screening setting (where unlabeled pairs vastly outnumber")
lines.append("true interactions).")
lines.append("")

lines.append("Per-seed sample sizes:")
for r in all_results:
    lines.append(f"  seed {r['seed']}: train={r['n_train']} ({r['n_train_pos']} pos / {r['n_train_neg']} unlabeled), "
                  f"test={r['n_test']} ({r['n_test_pos']} pos / {r['n_test_neg']} unlabeled)")
lines.append("")

lines.append("Bootstrap 95% CI for ROC-AUC (resampled test rows, 2000 iterations per seed):")
for r in all_results:
    lines.append(f"  seed {r['seed']}: XGBoost {r['xgb']['roc_auc']:.4f} [{r['xgb_ci'][0]:.4f}, {r['xgb_ci'][1]:.4f}]  "
                  f"RF {r['rf']['roc_auc']:.4f} [{r['rf_ci'][0]:.4f}, {r['rf_ci'][1]:.4f}]")
lines.append("")

for metric_key, label in [("roc_auc", "ROC-AUC"), ("auprc", "AUPRC")]:
    for model_key, model_label in [("xgb", "XGBoost"), ("rf", "Random Forest")]:
        mean, std = agg([model_key, metric_key])
        lines.append(f"{model_label} {label}: mean={mean:.4f}  std={std:.4f}  "
                      f"(per-seed: {[round(r[model_key][metric_key], 4) for r in all_results]})")
lines.append("")

for metric_key, label in [("p_at_20", "Precision@20"), ("p_at_50", "Precision@50")]:
    for model_key, model_label in [("xgb", "XGBoost"), ("rf", "Random Forest")]:
        mean, std = agg([model_key, metric_key])
        lines.append(f"{model_label} {label}: mean={mean:.4f}  std={std:.4f}")
lines.append("")

prev_mean, prev_std = np.mean([r["test_prevalence"] for r in all_results]), np.std([r["test_prevalence"] for r in all_results])
lines.append(f"Test set prevalence: mean={prev_mean:.4f}  std={prev_std:.4f}")
lines.append("")

len_mean, len_std = np.mean([r["len_auc"] for r in all_results]), np.std([r["len_auc"] for r in all_results])
shuf_mean, shuf_std = np.mean([r["shuf_auc"] for r in all_results]), np.std([r["shuf_auc"] for r in all_results])
lines.append(f"Length-only control AUC: mean={len_mean:.4f}  std={len_std:.4f}")
lines.append(f"Shuffled-label control AUC: mean={shuf_mean:.4f}  std={shuf_std:.4f}")
lines.append("")

lines.append("Per-seed confusion matrices (XGBoost, TN/FP/FN/TP):")
for r in all_results:
    cm = r["xgb"]["cm"]
    lines.append(f"  seed {r['seed']}: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

report = "\n".join(lines)
print("\n" + "=" * 70)
print(report)

with open("phase2c_results.txt", "w") as f:
    f.write(report + "\n")

# ----------------------------------------------------------------------
# Structured metrics CSV: one row per (seed, model)
# ----------------------------------------------------------------------
metrics_rows = []
for r in all_results:
    for model_key, model_label in [("xgb", "XGBoost"), ("rf", "Random Forest")]:
        m = r[model_key]
        ci = r[f"{model_key}_ci"]
        metrics_rows.append({
            "seed": r["seed"], "model": model_label,
            "n_train": r["n_train"], "n_test": r["n_test"],
            "n_train_pos": r["n_train_pos"], "n_train_neg": r["n_train_neg"],
            "n_test_pos": r["n_test_pos"], "n_test_neg": r["n_test_neg"],
            "roc_auc": m["roc_auc"], "roc_auc_ci_lo": ci[0], "roc_auc_ci_hi": ci[1],
            "auprc": m["auprc"], "p_at_20": m["p_at_20"], "p_at_50": m["p_at_50"],
            "f1": m["f1"], "accuracy": m["accuracy"],
            "tn": m["cm"][0, 0], "fp": m["cm"][0, 1], "fn": m["cm"][1, 0], "tp": m["cm"][1, 1],
        })
    metrics_rows.append({
        "seed": r["seed"], "model": "length_only", "roc_auc": r["len_auc"],
        "n_train": r["n_train"], "n_test": r["n_test"],
    })
    metrics_rows.append({
        "seed": r["seed"], "model": "shuffled_label", "roc_auc": r["shuf_auc"],
        "n_train": r["n_train"], "n_test": r["n_test"],
    })
pd.DataFrame(metrics_rows).to_csv("phase2c_metrics_by_seed.csv", index=False)

# ----------------------------------------------------------------------
# Average feature importance across seeds (+ per-seed columns)
# ----------------------------------------------------------------------
importance_df = pd.DataFrame({f"seed_{r['seed']}": r["importances"] for r in all_results})
importance_df["mean_importance"] = importance_df.mean(axis=1)
importance_df = importance_df.sort_values("mean_importance", ascending=False)
importance_df.to_csv("feature_importance_fixed.csv")

# ----------------------------------------------------------------------
# Test predictions and labels, for independent CI recomputation / auditing
# ----------------------------------------------------------------------
pred_rows = []
for r in all_results:
    for i in range(len(r["y_test"])):
        pred_rows.append({
            "seed": r["seed"], "id_a": r["test_id_a"][i], "id_b": r["test_id_b"][i],
            "label": r["y_test"][i], "xgb_proba": r["xgb_proba"][i], "rf_proba": r["rf_proba"][i],
        })
pd.DataFrame(pred_rows).to_csv("phase2c_test_predictions.csv", index=False)

# ----------------------------------------------------------------------
# Train/test pair assignments per seed (protein IDs only, not sequences --
# file size stays reasonable since raw sequences are excluded)
# ----------------------------------------------------------------------
split_rows = []
for r in all_results:
    for i in range(len(r["train_id_a"])):
        split_rows.append({"seed": r["seed"], "split": "train", "id_a": r["train_id_a"][i], "id_b": r["train_id_b"][i]})
    for i in range(len(r["test_id_a"])):
        split_rows.append({"seed": r["seed"], "split": "test", "id_a": r["test_id_a"][i], "id_b": r["test_id_b"][i]})
pd.DataFrame(split_rows).to_csv("phase2c_split_assignments.csv", index=False)

print("\nDone. Outputs written:")
print("  phase2c_results.txt              -- human-readable summary")
print("  phase2c_metrics_by_seed.csv       -- structured metrics, one row per seed/model")
print("  feature_importance_fixed.csv      -- per-seed + mean feature importance")
print("  phase2c_test_predictions.csv      -- per-row test predictions and labels")
print("  phase2c_split_assignments.csv     -- train/test pair assignments per seed")
print("\nTreat these numbers as the current reference point -- still preliminary")
print("given the modest sample size, but no longer subject to the split-leakage")
print("or negative-contamination bugs found in Phase 2b, and now covered by")
print("explicit sanity checks (protein overlap, negative contamination, duplicate")
print("pairs, swap-invariance, single-class splits) that fail loudly if violated.")
