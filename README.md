# Human Protein Interaction Prediction

An evaluation of whether simple, symmetric sequence-derived physicochemical features contain generalizable signal for distinguishing known human-human physical interactions (BioGRID) from randomly sampled non-observed pairs, under a protein-disjoint, leakage-controlled evaluation.

## Overview

This is not a deployable PPI predictor. The scientific question this project answers is:

> Do simple, symmetric sequence-derived physicochemical features contain generalizable signal for distinguishing known human-human physical interactions from randomly sampled non-observed pairs, when evaluation leakage is properly controlled?

**Answer: no — not from this feature set, under this evaluation.** Three phases were run, each fixing evaluation problems found in the previous one:

| Phase | ROC-AUC | Status | Problem |
|---|---|---|---|
| **Phase 2** | 0.96 | ❌ Invalid | Synthetic negatives (shuffled feature columns) + non-disjoint split. Model was learning shuffle artifacts, not biology. |
| **Phase 2b** | ~0.60 | ❌ Still invalid | Real negatives and symmetric features, but the protein-disjoint split logic (`min(id_a, id_b)`) was broken — 354 proteins leaked across train/test — and negative sampling only checked against the 968-pair sample, not the full BioGRID release (46 sampled "negatives" were real interactions elsewhere in BioGRID). |
| **Phase 2c** | **~0.53, 95% CI spans 0.5 on every seed** | ✅ Corrected | True protein-partition split, canonical pairs checked against the full filtered release, negatives sampled separately within each split, bootstrap CIs, and explicit sanity checks (protein overlap, negative contamination, duplicate pairs, swap-invariance, single-class splits) that fail loudly if violated. |

**This finding — no statistically reliable predictive signal from these 12 handcrafted features — is the actual result of the project, not a failure state.** It should not be read as "protein sequences contain no PPI information in general." It says these particular coarse physicochemical summary statistics (length, molecular weight, aromaticity, instability, GRAVY, isoelectric point — as symmetric sum/absdiff pairs) don't carry it, at this sample size, under this evaluation.

## Phase 2c Results (current, corrected)

5 seeds (42, 7, 123, 2024, 99), each with an independent protein partition, negatives resampled per split, and 2000-iteration bootstrap CIs per seed.

| Model | ROC-AUC (mean ± std) | AUPRC (mean ± std) | Precision@20 | Precision@50 |
|-------|---|---|---|---|
| XGBoost | 0.532 ± 0.057 | 0.558 ± 0.042 | 0.530 | 0.516 |
| Random Forest | 0.531 ± 0.072 | 0.555 ± 0.044 | 0.520 | 0.524 |

**Every seed's 95% bootstrap CI for ROC-AUC contains 0.5**, for both models. Full per-seed CIs are in `results/phase2c_results.txt`.

**Control baselines** (same range as the real models — this is the key evidence of no signal, not a caveat):

| Control | ROC-AUC (mean ± std) | Interpretation |
|---|---|---|
| Length-only model | 0.463 ± 0.052 | Not meaningfully different from the full 12-feature model |
| Shuffled-label model | 0.474 ± 0.036 | Near 0.5 as expected — confirms no remaining pipeline leakage |

**Note on AUPRC:** the test set is constructed 1:1 positive:unlabeled by design, so the no-skill AUPRC baseline here is ~0.5. This does not estimate precision under realistic biological screening prevalence, where non-interacting pairs vastly outnumber true interactions.

**Note on negatives:** all "negative" pairs in this project are **unlabeled/non-observed pairs** — randomly sampled protein pairs absent from the full filtered BioGRID release — not confirmed negatives. Absence of evidence in BioGRID is not proof of non-interaction.

**Sample size:** protein-disjoint splitting is expensive in pairs. With ~20% of proteins held out for test, only pairs where *both* proteins land in the same partition are kept (~300 of ~1200 pairs dropped per seed as boundary pairs). This is expected and correct — it is not a bug to fix by returning to random row splits, which would reintroduce protein leakage.

## Earlier Phases (superseded — kept for comparison and transparency)

### Phase 2b: 0.60 ROC-AUC — still invalid

| Model | ROC-AUC | AUPRC | F1 | Accuracy |
|-------|---------|-------|-----|----------|
| XGBoost (fixed) | 0.5997 | 0.5803 | 0.5785 | 57.84% |
| Random Forest (fixed) | 0.5741 | 0.5755 | 0.5512 | 55.15% |

This looked like a real, if modest, result, but a review found the protein-disjoint split was broken (`split_group = min(id_a, id_b)` does not guarantee disjointness — a protein can be the `min` in one pair and the `max` in another, leaking it across both splits) and negative sampling only excluded the 968-pair sample rather than the full BioGRID release. Phase 2c fixes both.

### Phase 2: 0.96 ROC-AUC — invalid

| Model | ROC-AUC | F1 Score | Accuracy |
|-------|---------|----------|----------|
| XGBoost | 0.9595 | 0.8973 | 89.75% |
| Random Forest | 0.7928 | 0.7210 | 72.25% |

Negatives were generated by shuffling feature columns within the positive set (not real non-interacting pairs), and the train/test split was random, not protein-disjoint. Top features were dominated by Protein-A-specific columns, which was itself a red flag given PPIs are symmetric (A-B ≡ B-A). The model was almost certainly learning artifacts of the shuffling process rather than biology.

## Features

### Phase 2c / 2b: symmetric features (12 total, current)

For each of `len`, `mw`, `arom`, `inst`, `gravy`, `pi`: a `_sum` and `_absdiff` feature — `f(A) + f(B)` and `|f(A) - f(B)|` — both invariant to swapping Protein A and B. Verified by an automated swap-invariance check in the Phase 2c pipeline (max drift < 1e-6 required, or the run fails).

### Phase 1: raw features (62 total, superseded)

Sequence length, 20-amino-acid composition fractions, 5 physicochemical properties, all computed separately per protein (`_a`/`_b` suffix) rather than symmetrically — this asymmetry was one of the problems Phase 2c fixes.

## Data

**Source:** BioGRID (Biological General Repository for Interaction Datasets), release **5.0.253**.
**Link:** [BioGRID Release Archive](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.253/BIOGRID-ALL-5.0.253.tab3.zip)

### Reproducing the data

The large BioGRID files are **not committed to this repo**. To reproduce:

1. Download `BIOGRID-ALL-5.0.253.tab3.csv` from the link above.
2. Run the filtering script:
   ```bash
   python scripts/filter_biogrid.py --input BIOGRID-ALL-5.0.253.tab3.csv --output biogrid_filtered.csv
   ```
   Filters applied: `Experimental System Type == "physical"`, both `Organism ID Interactor A/B == 9606` (human). Expected result: **1,214,422** human-human physical evidence rows.
3. This also produces a 1000-row evidence sample (`biogrid_filtered_sample_1000.csv`, `random_state=42`) — sequences are then retrieved separately (not part of this script) to produce `biogrid_sample_1000_with_sequences.csv`.
4. Place both `biogrid_filtered.csv` and `biogrid_sample_1000_with_sequences.csv` in the project root before running the modeling scripts.

**Important:** the 1000-row sample is drawn from *evidence rows*, not unique canonical pairs, so it can contain duplicate/reversed (A-B / B-A) pairs. The Phase 2c pipeline canonicalizes and deduplicates this automatically (2 such pairs were found and removed in the reference run) — do not deduplicate at the filtering stage, since the pipeline needs to do this consistently with its own protein-partition logic.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/harshitchaturvedy1/human-protein-interaction-prediction.git
cd human-protein-interaction-prediction

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Data preparation (required first)

```bash
python scripts/filter_biogrid.py --input BIOGRID-ALL-5.0.253.tab3.csv --output biogrid_filtered.csv
```
See [Reproducing the data](#reproducing-the-data) above for the full sequence-retrieval step.

### Phase 2c: Corrected Protein-Disjoint Evaluation (current)

```bash
python scripts/03c_phase2c_corrected_evaluation.py
```

**Inputs:** `biogrid_sample_1000_with_sequences.csv`, `biogrid_filtered.csv` (both in project root)
**Outputs:**
- `phase2c_results.txt` — human-readable summary
- `phase2c_metrics_by_seed.csv` — structured metrics, one row per seed/model
- `feature_importance_fixed.csv` — per-seed + mean feature importance
- `phase2c_test_predictions.csv` — per-row test predictions and labels (for independent CI recomputation)
- `phase2c_split_assignments.csv` — train/test protein-pair assignments per seed

**Runtime:** ~5-10 minutes (5 seeds, full pipeline each)
**Requires:** pandas, numpy, scikit-learn, xgboost, biopython

**Built-in sanity checks** (the run fails loudly, not silently, if any of these are violated):
- No protein appears in both final train and test data
- No sampled "negative" pair is actually a known BioGRID interaction
- No duplicate or reversed pair within a split
- The exact requested number of negatives is generated
- Swapping Protein A and B does not change the feature vector
- Neither split contains only one class

### Phase 1 & Phase 2b (superseded, kept for reference/comparison only)

```bash
python scripts/01_phase1_feature_engineering.py   # -> biogrid_features_phase1.csv
python scripts/02_phase2_baseline_models.py        # -> Phase 2's invalid 0.96 result
python scripts/03_phase2b_fixed_evaluation.py       # -> Phase 2b's still-invalid 0.60 result
```
Do not use these for any actual conclusion — see the phase comparison table above for why.

## Project Structure

```
human-protein-interaction-prediction/
├── scripts/
│   ├── filter_biogrid.py
│   ├── 01_phase1_feature_engineering.py
│   ├── 02_phase2_baseline_models.py
│   ├── 03_phase2b_fixed_evaluation.py
│   └── 03c_phase2c_corrected_evaluation.py
├── results/
│   ├── feature_importance.csv               (Phase 2, superseded)
│   ├── phase2_results.txt                   (Phase 2, superseded)
│   ├── feature_importance_fixed.csv          (Phase 2b, superseded — Phase 2c overwrites this filename with per-seed data)
│   ├── phase2b_results.txt                   (Phase 2b, superseded)
│   ├── phase2c_results.txt
│   ├── phase2c_metrics_by_seed.csv
│   ├── phase2c_test_predictions.csv
│   └── phase2c_split_assignments.csv
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

## Key Findings

✅ **Phase 2c evaluation is trustworthy** — protein-disjoint split verified on final data (not a tautological pre-check), canonical known-pairs checked against the full BioGRID release, explicit sanity checks fail loudly on violation, bootstrap CIs reported

❌ **No statistically reliable signal found** — every seed's 95% CI for ROC-AUC contains 0.5, for both models; neither model meaningfully outperforms the length-only or shuffled-label controls

⚠️ **This is a negative result about these 12 features under this evaluation, not a claim that protein sequences carry no PPI information** — see Next Steps for what a positive result would likely require

✅ **Fully auditable** — per-row test predictions, per-seed split assignments, and per-seed feature importance are all saved, so results can be independently re-verified without rerunning the full pipeline

## Next Steps (future work — not required to close out this phase)

- Use a much larger, deduplicated positive dataset with sequences (current test sets are ~60-90 pairs per seed after protein-disjoint filtering — CIs are wide)
- Use degree-, length-, or localization-matched non-observed pairs instead of uniform random sampling, for harder negatives
- Add sequence-cluster-disjoint evaluation (stricter than protein-disjoint)
- Test protein language-model embeddings (ESM-2) in place of hand-engineered statistics
- Pivot to BioGRID evidence-quality prediction, which avoids constructing a negative-interaction class at all

Neural networks, transformers, ESM embeddings, and further model tuning are explicitly **not** planned for the current baseline — they would add complexity without addressing the actual limiting factors (sample size, evaluation rigor).

## Limitations

1. Negative sampling uses unlabeled/non-observed pairs, not confirmed negatives — absence from BioGRID is not proof of non-interaction
2. Small sample size (~950 unique positive pairs before splitting; ~60-90 test pairs per seed after protein-disjoint filtering)
3. Binary classification only (interaction / no interaction)
4. Validation needed on other PPI databases
5. Protein-disjoint splitting substantially reduces usable pair count compared to random splitting — this is a necessary cost of valid evaluation, not something to "fix" by relaxing the split

## License

## License

GPL-3.0 License — see the [LICENSE](LICENSE) file for details.


## Author

Harshit Chaturvedy

## Citation

```bibtex
@project{chaturvedy2026ppi,
  title={Human Protein Interaction Prediction: An Evaluation of Symmetric Sequence-Derived Features under Protein-Disjoint Splitting},
  author={Chaturvedy, Harshit},
  year={2026},
  url={https://github.com/harshitchaturvedy1/human-protein-interaction-prediction}
}
```

## Contact

- Email: harshitchaturvedy@gmail.com, saisohan.panda@gmail.com, shivanshsahni10@gmail.com

---

**Status:** ✅ Phase 2c correction complete — evaluation is leakage-checked and sanity-asserted; result is a validated negative finding (no signal from these 12 features under protein-disjoint evaluation), not a deployable model
**Last Updated:** July 2026
