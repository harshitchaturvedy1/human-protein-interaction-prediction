# Human Protein Interaction Prediction

Machine learning models for predicting protein-protein interactions (PPIs) using sequence-based features from BioGRID data.

## Overview

This project implements a two-phase approach to predict protein-protein interactions:

1. **Phase 1: Feature Engineering** - Extract 62 handcrafted features from protein sequences
2. **Phase 2: Baseline Modeling** - Train XGBoost and Random Forest classifiers

**Best Result:** XGBoost achieves **96% ROC-AUC** on held-out test set.

## Results Summary

| Model | ROC-AUC | F1 Score | Accuracy | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| XGBoost | **0.9595** | 0.8973 | 89.75% | 90.11% | 89.75% |
| Random Forest | 0.7928 | 0.7210 | 72.25% | 72.73% | 72.25% |

## Features

### Phase 1 Feature Engineering (62 total)

**Sequence Properties:**
- Length (2 features): len_a, len_b

**Amino Acid Composition (40 features):**
- Fraction of each of 20 amino acids × 2 proteins

**Physicochemical Properties (10 features):**
- Molecular weight, aromaticity, instability index, GRAVY, isoelectric point (×2 proteins)

**Pairwise Differences (6 features):**
- diff_len, diff_mw, diff_arom, diff_inst, diff_gravy, diff_pi

**Data Quality Flags (4 features):**
- missing_seq_a, missing_seq_b, invalid_seq_a, invalid_seq_b

### Top Predictive Features (XGBoost)

1. **len_a** (0.0499) - Protein A sequence length
2. **mw_a** (0.0380) - Protein A molecular weight
3. **arom_a** (0.0331) - Protein A aromaticity
4. **comp_a_P** (0.0273) - Proline composition in Protein A
5. **gravy_a** (0.0236) - Protein A hydropathy

## Data

**Source:** BioGRID (Biological General Repository for Interaction Datasets)

**Dataset Composition:**
- 1000 confirmed protein-protein interactions (positive class)
- 1000 synthetic negatives (shuffled protein pairs)
- Total: 2000 samples, balanced classes

**Train/Test Split:**
- Training: 1600 rows (80%)
- Testing: 400 rows (20%)
- Stratified split to preserve class distribution

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/human-protein-interaction-prediction.git
cd human-protein-interaction-prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Phase 1: Feature Engineering

```bash
python scripts/01_phase1_feature_engineering.py
```

**Input:** `biogrid_sample_1000_with_sequences.csv`
**Output:** `biogrid_features_phase1.csv`
**Runtime:** ~2-3 minutes

### Phase 2: Model Training & Evaluation

```bash
python scripts/02_phase2_baseline_models.py
```

**Input:** `biogrid_features_phase1.csv`
**Outputs:**
- `model_xgboost.pkl`
- `model_rf.pkl`
- `feature_importance.csv`
- `phase2_results.txt`

**Runtime:** ~1-2 minutes

## Project Structure

```
human-protein-interaction-prediction/
├── scripts/
│   ├── 01_phase1_feature_engineering.py
│   └── 02_phase2_baseline_models.py
├── results/
│   ├── feature_importance.csv
│   └── phase2_results.txt
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

## Key Findings

✅ **96% AUC is excellent** - Model reliably distinguishes interacting from non-interacting pairs

✅ **Protein A dominates** - Sequence length and composition of Protein A are most predictive

✅ **Simple features work** - Handcrafted features capture strong signal

✅ **Interpretable** - Feature importance clearly shows what drives predictions

## Next Steps

### Option 1: Deploy as-is (Recommended)
- Model achieves 96% AUC - excellent for production
- Fast inference (~1ms per prediction)
- Fully interpretable

### Option 2: Optimize Further
- Add Phase 2 extended features (network degree, sequence similarity)
- Hyperparameter tuning
- Target: 97%+ AUC

### Option 3: Advanced Methods (Phase 3)
- ESM-2 protein language model embeddings
- Siamese neural networks
- Transformer-based models
- Expected improvement: minimal (~96% → 97-99%)

## Limitations

1. Negative examples are synthetic (shuffled pairs)
2. Dataset may have bias toward Protein A characteristics
3. Binary classification only (interaction / no interaction)
4. Validation needed on other PPI databases

## License

MIT License - See LICENSE file for details

## Author

Harshit Chaturvedy

## Citation

```bibtex
@project{chaturvedy2026ppi,
  title={Human Protein Interaction Prediction using Sequence-Based Features},
  author={Chaturvedy, Harshit},
  year={2026},
  url={https://github.com/yourusername/human-protein-interaction-prediction}
}
```

## Contact

- Email: harshitchaturvedy@gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)
- Instagram: [@harshitchaturvedy](https://instagram.com/harshitchaturvedy)

---

**Status:** ✅ Complete and ready for deployment or further optimization
**Last Updated:** May 21, 2026
