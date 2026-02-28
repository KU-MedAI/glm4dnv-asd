# Challenges in De Novo Variant-Based Autism Spectrum Disorder Prediction Using Genomic Language Models

## Abstract
<div align="center">
<img width="2009" height="1520" alt="260122 figure1 overview   Model architecture" src="https://github.com/user-attachments/assets/6bfbe945-7285-4fcc-b36a-9190c0eec72b" />

</div>
<br>
<br>

Background: Genomic language models (gLMs) have demonstrated strong performance in sequence-level tasks such as variant pathogenicity classification and regulatory activity prediction. However, their utility for disease prediction at the sample level remains largely unexplored. Systematic benchmarking of diverse gLM architectures within a unified disease prediction framework is still lacking, leaving it unclear which representation strategies are most informative for clinical application and whether task-specific fine-tuning can bridge the gap between sequence-level annotation and sample-level disease prediction. Here, we address this gap using autism spectrum disorder (ASD) as a representative complex disease model. 

Results: To assess the utility of gLM embeddings in complex disease prediction, we evaluated their performance in classifying ASD risk. Our findings indicate that task-specific fine-tuning did not yield a statistically significant improvement in overall predictive accuracy. Nonetheless, a detailed analysis of the embeddings and attention scores demonstrated that the fine-tuning process elicited significant structural modifications within the embedding space and enabled a strategic redistribution of attention scores. This phenomenon was notably observed in non-coding and regulatory elements relevant to disease pathogenesis.

Conclusions: Although fine-tuning was applied to integrate functional variant information, this approach did not directly translate into improved predictive performance, highlighting a fundamental gap between variant-level representation learning and sample-level disease prediction in current gLM frameworks. Nevertheless, these representational shifts demonstrate that fine-tuning can reshape gLM embedding geometry around variant positions, even when this does not yet translate to improved sample-level classification. Furthermore, by systematically benchmarking diverse gLM architectures within a unified disease prediction pipeline, this study establishes a critical baseline and identifies key gaps—particularly in variant-aware pretraining and disease-specific representation learning—that must be addressed to advance gLMs toward clinical utility.
