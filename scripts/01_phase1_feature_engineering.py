import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
warnings.filterwarnings('ignore')

print("Step 1: Loading BioGRID data...")
df = pd.read_csv('biogrid_sample_1000_with_sequences.csv')
print(f"  Loaded {len(df)} rows")

if 'Sequence Interactor A' in df.columns:
    df['Sequence A'] = df['Sequence Interactor A']
if 'Sequence Interactor B' in df.columns:
    df['Sequence B'] = df['Sequence Interactor B']

print("\nStep 2: Computing sequence lengths...")
df['len_a'] = df['Sequence A'].str.len()
df['len_b'] = df['Sequence B'].str.len()
print(f"  Mean length A: {df['len_a'].mean():.1f}, Mean length B: {df['len_b'].mean():.1f}")

print("\nStep 3: Computing amino acid composition...")
aa_list = 'ACDEFGHIKLMNPQRSTVWY'

def get_aa_composition(seq):
    if pd.isna(seq) or len(seq) == 0:
        return {aa: 0.0 for aa in aa_list}
    seq = seq.upper()
    total = len(seq)
    return {aa: seq.count(aa) / total for aa in aa_list}

comp_a = df['Sequence A'].apply(get_aa_composition)
for aa in aa_list:
    df[f'comp_a_{aa}'] = comp_a.apply(lambda x: x[aa])

comp_b = df['Sequence B'].apply(get_aa_composition)
for aa in aa_list:
    df[f'comp_b_{aa}'] = comp_b.apply(lambda x: x[aa])

print(f"  Added {2 * len(aa_list)} composition features")

print("\nStep 4: Computing physicochemical properties (this may take 1-2 min)...")

def get_protparam(seq):
    if pd.isna(seq) or len(seq) < 3:
        return {
            'mw': np.nan,
            'arom': np.nan,
            'inst': np.nan,
            'gravy': np.nan,
            'pi': np.nan
        }
    try:
        seq = str(seq).upper()
        pp = ProteinAnalysis(seq)
        return {
            'mw': pp.molecular_weight(),
            'arom': pp.aromaticity(),
            'inst': pp.instability_index(),
            'gravy': pp.gravy(),
            'pi': pp.isoelectric_point()
        }
    except Exception as e:
        return {
            'mw': np.nan,
            'arom': np.nan,
            'inst': np.nan,
            'gravy': np.nan,
            'pi': np.nan
        }

print("  Processing Protein A...")
protparam_a = df['Sequence A'].apply(get_protparam)
df['mw_a'] = protparam_a.apply(lambda x: x['mw'])
df['arom_a'] = protparam_a.apply(lambda x: x['arom'])
df['inst_a'] = protparam_a.apply(lambda x: x['inst'])
df['gravy_a'] = protparam_a.apply(lambda x: x['gravy'])
df['pi_a'] = protparam_a.apply(lambda x: x['pi'])

print("  Processing Protein B...")
protparam_b = df['Sequence B'].apply(get_protparam)
df['mw_b'] = protparam_b.apply(lambda x: x['mw'])
df['arom_b'] = protparam_b.apply(lambda x: x['arom'])
df['inst_b'] = protparam_b.apply(lambda x: x['inst'])
df['gravy_b'] = protparam_b.apply(lambda x: x['gravy'])
df['pi_b'] = protparam_b.apply(lambda x: x['pi'])

print(f"  Added 10 physicochemical features")

print("\nStep 5: Computing pairwise differences...")
df['diff_len'] = abs(df['len_a'] - df['len_b'])
df['diff_mw'] = abs(df['mw_a'] - df['mw_b'])
df['diff_arom'] = abs(df['arom_a'] - df['arom_b'])
df['diff_inst'] = abs(df['inst_a'] - df['inst_b'])
df['diff_gravy'] = abs(df['gravy_a'] - df['gravy_b'])
df['diff_pi'] = abs(df['pi_a'] - df['pi_b'])

print(f"  Added 6 pairwise difference features")

print("\nStep 6: Creating missing data indicators...")
df['missing_seq_a'] = df['Sequence A'].isna().astype(int)
df['missing_seq_b'] = df['Sequence B'].isna().astype(int)
df['invalid_seq_a'] = (df['len_a'] < 3).astype(int)
df['invalid_seq_b'] = (df['len_b'] < 3).astype(int)

missing_pct = (df['missing_seq_a'].sum() + df['missing_seq_b'].sum()) / (2 * len(df)) * 100
print(f"  Missing sequences: {missing_pct:.2f}%")
print(f"  Added 4 missing data flags")

print("\nStep 7: Cleaning up and saving...")

df[['mw_a', 'mw_b', 'arom_a', 'arom_b', 'inst_a', 'inst_b', 'gravy_a', 'gravy_b', 'pi_a', 'pi_b']] = \
    df[['mw_a', 'mw_b', 'arom_a', 'arom_b', 'inst_a', 'inst_b', 'gravy_a', 'gravy_b', 'pi_a', 'pi_b']].fillna(
        df[['mw_a', 'mw_b', 'arom_a', 'arom_b', 'inst_a', 'inst_b', 'gravy_a', 'gravy_b', 'pi_a', 'pi_b']].mean()
    )

feature_cols = [
    'Official Symbol Interactor A', 'Official Symbol Interactor B',
    'SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B',
    'Experimental System', 'Experimental System Type'
]
feature_cols += [col for col in df.columns if any([
    col.startswith('len_'),
    col.startswith('comp_'),
    col.startswith('mw_'),
    col.startswith('arom_'),
    col.startswith('inst_'),
    col.startswith('gravy_'),
    col.startswith('pi_'),
    col.startswith('diff_'),
    col.startswith('missing_'),
    col.startswith('invalid_')
])]

df_features = df[[col for col in feature_cols if col in df.columns]].copy()

output_file = 'biogrid_features_phase1.csv'
df_features.to_csv(output_file, index=False)

print(f"\nSaved {len(df_features)} rows x {len(df_features.columns)} columns to '{output_file}'")
print(f"\nTotal features: {len([col for col in df_features.columns if col not in ['Official Symbol Interactor A', 'Official Symbol Interactor B', 'SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B', 'Experimental System', 'Experimental System Type']])}")

print("\n" + "="*80)
print("Phase 1 Complete! Ready for Phase 2 (baseline modeling).")
print("="*80)
