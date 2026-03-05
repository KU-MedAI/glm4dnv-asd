# Zero-shot

## Overview

This directory contains code for variant pooling to derive **variant-level representations** from genomic language model embeddings.

- Extract token embeddings corresponding to the **mutated nucleotide positions** from model outputs.
- If a variant affects **multiple positions** (e.g., frameshift mutations), collect embeddings from all affected loci.
- Apply **max pooling** across these embeddings to obtain a single variant-level feature vector.
- Use the pooled vector as the **final representation for each variant**.

This pooling procedure is applied to **de novo variant datasets used for downstream ASD prediction**.

## Usage

Run this notebook: `zs_variant_pooling.ipynb`
