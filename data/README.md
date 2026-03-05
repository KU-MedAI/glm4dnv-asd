# Data

---

This study uses two types of datasets: **de novo variant datasets for ASD prediction** and **datasets used for fine-tuning tasks**.

## 1. De novo variant datasets -> 어떻게 생겼는지 설명 추가

De novo variants were collected from multiple ASD cohorts:

- SSC  
- SPARK  
- MSSNG  
- Korean ASD cohort  

In addition to genomic variants, **clinical severity annotations** were obtained from the same cohorts when available. These annotations were used for downstream analyses of clinical severity.

## 2. Fine-tuning task datasets -> 어떻게 생겼는지 설명 추가
-> 우리 실제 이름으로 변경 필요
<br> 

Fine-tuning tasks were constructed using publicly available datasets:

- ClinVar  
- gnomAD  
- BEND benchmark  
- BRAIN-MAGNET model (NCRE activity dataset)

All datasets were processed to ensure consistent input construction, and sequence inputs were generated from the **GRCh38 reference genome**.


## Data Access
- De novo variant data
  - Due to data access restrictions, raw datasets cannot be distributed in this repository.
- Fine-tuning task data
  - 출처 각각 기입하기
