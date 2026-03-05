# Zero-shot

## Model List

- **DNABERT**
- **DNABERT-2**
- **Nucleotide Transformer V2**
- **Nucleotide Transformer V3**
- **HyenaDNA**
- **Evo 2**
- **PhyloGPN**

## Variant Pooling

To obtain variant-level representations from sequence embeddings, token embeddings corresponding to the mutated nucleotide positions were extracted from the model outputs. For variants affecting multiple positions (e.g., frameshift mutations), the embeddings at the affected loci were aggregated using max pooling to produce a single variant-level feature vector. This pooled representation was used as the final embedding for each variant.
