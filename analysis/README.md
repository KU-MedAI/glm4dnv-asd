# Analysis

---

## Overview
This section contains analyses designed to investigate how genomic language model (gLM) representations change after fine-tuning in the context of Autism Spectrum Disorder (ASD) prediction.

## Directory Structure
```text
analysis/
├── README.md
├── fine-tuning-effects/
│   ├── embedding_vector_analysis.ipynb
│   └── attention_score_analysis.ipynb
│
└── biological-interpretability/
    ├── attn_score_based_enrichment.ipynb
    └── attn_scroe_with_severity_enrichment.ipynb
```

---

## Analysis of fine-tuning effects
### Representational shifts in latent embedding space

- 1. Cosine similarity distribution analysis
  - Normalize reference and variant embeddings and calculate cosine similarity.
  - Compare Zero-shot vs. Fine-tuned distributions using histogram grids.
  - Quantify distributional shifts using Cliff's Delta (δ) effect size.

- 2. VEP-based variant impact enrichment analysis
  - Categorize variants into HIGH and NON-HIGH impact groups based on VEP annotations.
  - Calculate Delta Cosine (Sim_FT − Sim_Zero) for each variant.
  - Compare shifts between impact groups to assess molecular consequence sensitivity.

- 3. Pathogenicity enrichment analysis based on CADD scores
  - Define Pathogenic and Benign subgroups using CADD Phred score thresholds.
  - Perform Mann–Whitney U tests with Bonferroni correction for statistical significance.
  - Compare Delta Cosine patterns to validate the clinical interpretability of model shifts.

### Functional variant enrichment in attention scores
### Alignment between prediction confidence and disease gene prioritization

---

## Biological interpretability of ASD prediction
This analysis assesses whether the mutations prioritized by the model through attention scores converge on biological pathways associated with ASD

### Attention-based variant prioritization enrichment analysis
- Normalize attention scores within each sample using CLR transformation.
- Select top and bottom 10% attention-ranked variants.
- Map variants to genes and perform GO Biological Process enrichment analysis.

### Enrichment analysis of ASD subgroups based on severity annotation
- Apply the same enrichment framework to clinically defined ASD subgroups.
- Define severe groups using ADOS and VABS severity annotations.
- Compare pathway enrichment patterns of top/bottom attention-ranked variants within each subgroup.
