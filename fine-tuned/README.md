# Fine-Tuning Code

---

## Overview
This directory contains the code and resources for **{short description of the module}** in the benchmarking pipeline.

## Directory Structure
```text
fine-tuned/
├── README.md
└── {model directories}
     ├── classification
     │    ├── ft_{model}_classification.py
     │    └── run_ft_{model}_classification.sh
     │
     ├── regression
     │    ├── ft_{model}_regression.py
     │    └── run_ft_{model}_regression.sh
     │
     └── variant-pooling
          ├── ft_{model}_variant_pooling.py
          └── run_ft_{model}_variant_pooling.sh
```

## Tasks

Two types of fine-tuning tasks are implemented:

- **Classification**
- **Regression**

Each task is implemented separately for each genomic language model.

## Usage

Example:

```bash
bash DNABERT/classification/run_ft_dnabert_classification.sh
```

After fine-tuning, variant pooling is performed using scripts in: 

```bash
bash DNABERT/variant-pooling/run_ft_dnabert_variant_pooling.sh
```
