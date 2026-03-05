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

#### Cosine similarity distribution analysis
- Normalize reference and variant embeddings and compute cosine similarity.
- Compare **Zero-shot** and **Fine-tuned** distributions using histogram grids.
- Quantify distributional shifts using **Cliff's Delta (δ)** effect size.

#### VEP-based variant impact enrichment analysis
- Categorize variants into **HIGH** and **NON-HIGH** impact groups using VEP annotations.
- Compute **Delta Cosine (Sim_FT − Sim_Zero)** for each variant.
- Compare distributions between impact groups to evaluate sensitivity to molecular consequences.

#### Pathogenicity enrichment analysis based on CADD scores
- Define **Pathogenic** and **Benign** variants using CADD Phred score thresholds.
- Perform **Mann–Whitney U tests with Bonferroni correction**.
- Evaluate whether Delta Cosine shifts align with predicted pathogenicity.

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
