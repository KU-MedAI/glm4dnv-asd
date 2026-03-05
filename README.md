# Challenges in De Novo Variant-Based Autism Spectrum Disorder Prediction Using Genomic Language Models

---

## Abstract
<div align="center">
<img width="2009" height="1520" alt="260122 figure1 overview   Model architecture" src="https://github.com/user-attachments/assets/6bfbe945-7285-4fcc-b36a-9190c0eec72b" />

</div>
<br>
<br>

Background: Genomic language models (gLMs) have demonstrated strong performance in sequence-level tasks such as variant pathogenicity classification and regulatory activity prediction. However, their utility for disease prediction at the sample level remains largely unexplored. Systematic benchmarking of diverse gLM architectures within a unified disease prediction framework is still lacking, leaving it unclear which representation strategies are most informative for clinical application and whether task-specific fine-tuning can bridge the gap between sequence-level annotation and sample-level disease prediction. Here, we address this gap using autism spectrum disorder (ASD) as a representative complex disease model. 

Results: To assess the utility of gLM embeddings in complex disease prediction, we evaluated their performance in classifying ASD risk. Our findings indicate that task-specific fine-tuning did not yield a statistically significant improvement in overall predictive accuracy. Nonetheless, a detailed analysis of the embeddings and attention scores demonstrated that the fine-tuning process elicited significant structural modifications within the embedding space and enabled a strategic redistribution of attention scores. This phenomenon was notably observed in non-coding and regulatory elements relevant to disease pathogenesis.

Conclusions: Although fine-tuning was applied to integrate functional variant information, this approach did not directly translate into improved predictive performance, highlighting a fundamental gap between variant-level representation learning and sample-level disease prediction in current gLM frameworks. Nevertheless, these representational shifts demonstrate that fine-tuning can reshape gLM embedding geometry around variant positions, even when this does not yet translate to improved sample-level classification. Furthermore, by systematically benchmarking diverse gLM architectures within a unified disease prediction pipeline, this study establishes a critical baseline and identifies key gaps—particularly in variant-aware pretraining and disease-specific representation learning—that must be addressed to advance gLMs toward clinical utility.

---

## Overview

This repository provides the code and analysis pipeline used in our study  
**“Challenges in De Novo Variant-Based Autism Spectrum Disorder Prediction Using Genomic Language Models.”**

We systematically benchmark seven genomic language models (gLMs) for predicting **autism spectrum disorder (ASD)** from **de novo variants**.

This repository includes:

- a unified **benchmarking framework for genomic language models**
- a **sample-level ASD prediction pipeline**
- analyses of **fine-tuning effects on gLM representations**
- evaluation of the **biological interpretability of model attention**

---

## Benchmark Design

### Evaluated Genomic Language Models

Seven genomic language models were benchmarked:

- DNABERT  
- DNABERT-2  
- Nucleotide Transformer V2  
- Nucleotide Transformer V3  
- HyenaDNA  
- Evo 2  
- PhyloGPN  

Each model was evaluated under two settings:

- **Zero-shot representation**
- **Task-specific fine-tuning**

---

## Benchmark Pipeline

1. **Embedding Variant Sequences**  
- Variant-level embeddings are extracted using gLMs.  
- At this stage, variant pooling is performed by using only the tokens that contain variants.

2. **Disease (ASD) Prediction**  
- A Set Transformer model aggregates variant representations at the sample level to predict ASD status.

3. **Analysis**
- **Fine-tuning Effects Analysis**
  - Representation shifts in the gLM embedding space
  - Redistribution of attention scores from the Set Transformer
- **Biological Interpretability Analysis**
  - Enrichment of disease-relevant biological pathways

---

## Repository Structure

```bash
├── data
│   └── README.md
│
├── zero-shot
│   ├── README.md
│   └── zs_variant_pooling.ipynb
│
├── fine-tuned
│   ├── README.md
│   ├── ft_classification
│   │   └── {model directories}
│   │
│   ├── ft_regression
│   │   └── {model directories}
│   │
│   └── ft_variant_pooling
│       └── {model directories}
│
├── set_transformer
│   └── set_transformer.ipynb
│
└── analysis
    ├── fine-tuning-effects
    │   ├── README.md
    │   ├── embedding_analysis.ipynb
    │   └── untitle1.ipynb
    │   └── untitle2.ipynb
    │
    └── biological_interpretability
        ├── README.md
        ├── attention_only_enrichment_analysis.py
        └── severity_joint_enrichment_analysis.py
```

---

## Environment Setting
### Required Packages

```python
# PyTorch (CUDA support recommended)
pip install torch torchvision torchaudio

# Data Processing
pip install pandas numpy scipy

# Progress Display
pip install tqdm

# Analysis - GSEApy Enrichr
pip install gseapy

# Optional: Experiment Tracking
pip install wandb
```

---

## Reproducing the Benchmark

The main steps to reproduce the benchmark are outlined below.

### 1-1. Zero-shot variant representation

Run the notebook to extract variant embeddings using pretrained genomic language models.

```bash
zero-shot/zs_variant_pooling.ipynb
```

### 1-2. Fine-tuning genomic language models

Run task-specific fine-tuning for each model.

```bash
bash fine-tuned/ft_classification/{model}/run_sweep.sh
```

### 1-3. Variant pooling

Variant pooling for fine-tuned models is implemented in the following directory.

```bash
bash fine-tuned/variant_pooling/{model}/pooling_best.sh
```

### 2. Sample-level ASD prediction

Run the Set Transformer notebook to perform ASD classification using the representations generated from each task (zero-shot and fine-tuned).

```bash
set_transformer/set_transformer.ipynb
```

---

## Data

This study uses two types of datasets: **de novo variant datasets for ASD prediction** and **datasets used for fine-tuning tasks**.

### 1. De novo variant datasets

De novo variants were collected from multiple ASD cohorts:

- SSC  
- SPARK  
- MSSNG  
- Korean ASD cohort  

In addition to genomic variants, **clinical severity annotations** were obtained from the same cohorts when available. These annotations were used for downstream analyses of clinical severity.

Due to data access restrictions, raw datasets cannot be distributed in this repository.

Instructions for preparing the required data are provided in:  
`data/README.md`

### 2. Fine-tuning task datasets

Fine-tuning tasks were constructed using publicly available datasets:

- ClinVar  
- gnomAD  
- BEND benchmark  
- BRAIN-MAGNET model (NCRE activity dataset)

All datasets were processed to ensure consistent input construction, and sequence inputs were generated from the **GRCh38 reference genome**.

Detailed descriptions of the fine-tuning datasets and preprocessing procedures are provided in:  
`data/README.md`

---

## Analysis

The analysis scripts reproduce the main results reported in the paper, including:

Fine-tuning effects on gLM representations

Biological interpretability of ASD prediction

See the `analysis/` directory for details.
