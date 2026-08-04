"""
filter_biogrid.py
============================================================================
Reproducible data-construction script for this project's BioGRID input.

Starts from a named BioGRID release and applies the exact row-level
filters used to produce biogrid_filtered.csv and the 1000-row modeling
sample, biogrid_sample_1000_with_sequences.csv (sequences are added in a
separate retrieval step, not by this script).

BioGRID release used: BioGRID 5.0.253
Source file:          BIOGRID-ALL-5.0.253.tab3.csv
Download:              https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.253/BIOGRID-ALL-5.0.253.tab3.zip

Filters applied (in order):
  1. Experimental System Type == "physical"
  2. Organism ID Interactor A == 9606 (Homo sapiens)
  3. Organism ID Interactor B == 9606 (Homo sapiens)

No pair-level cleaning (self-pairs, canonicalization, deduplication) is
done here -- that happens downstream in the modeling pipeline
(03c_phase2c_corrected_evaluation.py), which needs full control over
canonicalization to keep it consistent with the protein-disjoint split.
This script only reproduces the original row-level filtering.

Expected result on BioGRID 5.0.253: 1,214,422 human-human physical
evidence rows. (Downstream scripts may report a slightly different
count of *unique canonical pairs*, since a single protein pair can have
multiple evidence rows in BioGRID -- e.g. from different publications.)

Columns retained (metadata not needed for this analysis is dropped --
author, publication source, throughput, score, modification,
qualifications, tags, ontology fields, synonyms, systematic names,
TrEMBL accessions, RefSeq accessions):
  - Interactor identifiers (BioGRID IDs)
  - Official symbols (Interactor A / B)
  - Experimental System
  - Experimental System Type
  - Source Database
  - Swiss-Prot accessions (Interactor A / B)
  - Organism IDs (Interactor A / B)
  - Organism names (Interactor A / B)

Output: biogrid_filtered.csv

Note on sampling: the original 1000-row sample
(biogrid_sample_1000_with_sequences.csv) was drawn from EVIDENCE ROWS
via df.sample(n=1000, random_state=42), not from unique canonical
protein pairs. This means the raw sample can and does contain duplicate
or reversed (A-B / B-A) pairs -- confirmed: 2 such pairs were found and
removed in the Phase 2c pipeline. Canonicalization and deduplication are
handled downstream, not by resampling here, so this script's output
matches what was actually used to generate the project's positive set.

Usage:
  python filter_biogrid.py --input BIOGRID-ALL-5.0.253.tab3.csv --output biogrid_filtered.csv
"""

import argparse
import pandas as pd

KEEP_COLUMNS = [
    "BioGRID ID Interactor A", "BioGRID ID Interactor B",
    "Official Symbol Interactor A", "Official Symbol Interactor B",
    "Experimental System", "Experimental System Type",
    "Source Database",
    "SWISS-PROT Accessions Interactor A", "SWISS-PROT Accessions Interactor B",
    "Organism ID Interactor A", "Organism ID Interactor B",
    "Organism Name Interactor A", "Organism Name Interactor B",
]

HUMAN_TAXID = 9606


def filter_biogrid(input_path: str, output_path: str, sample_n: int = 1000, sample_seed: int = 42):
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    n_total = len(df)
    print(f"  {n_total} total rows in raw release")

    # Filter 1: physical evidence only (excludes genetic interactions)
    df = df[df["Experimental System Type"].str.lower() == "physical"]
    print(f"  After Experimental System Type == 'physical': {len(df)} rows")

    # Filter 2 & 3: both interactors human
    df = df[
        (df["Organism ID Interactor A"] == HUMAN_TAXID) &
        (df["Organism ID Interactor B"] == HUMAN_TAXID)
    ]
    print(f"  After both organisms == {HUMAN_TAXID} (human): {len(df)} rows")

    # Keep only the columns needed downstream
    available_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = set(KEEP_COLUMNS) - set(available_cols)
    if missing:
        print(f"  WARNING: expected columns not found in input, skipped: {missing}")
    df = df[available_cols]

    df.to_csv(output_path, index=False)
    print(f"\nSaved filtered dataset to {output_path}: {len(df)} rows")

    # Modeling sample (evidence-row sample, not canonical-pair sample --
    # see note in module docstring)
    sample_df = df.sample(n=min(sample_n, len(df)), random_state=sample_seed)
    sample_path = output_path.replace(".csv", f"_sample_{sample_n}.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved {sample_n}-row evidence sample to {sample_path}")
    print("\nNOTE: this sample is drawn from evidence rows and may contain")
    print("duplicate or reversed (A-B/B-A) pairs. Downstream modeling code")
    print("canonicalizes and deduplicates before use.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BioGRID release to human-human physical interactions.")
    parser.add_argument("--input", required=True, help="Path to raw BIOGRID-ALL-*.tab3.csv")
    parser.add_argument("--output", default="biogrid_filtered.csv", help="Path to write filtered CSV")
    parser.add_argument("--sample-n", type=int, default=1000, help="Rows to sample for the modeling set")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    filter_biogrid(args.input, args.output, args.sample_n, args.sample_seed)
