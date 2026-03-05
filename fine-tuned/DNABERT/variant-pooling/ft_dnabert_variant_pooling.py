#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import argparse
import ast
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from peft import PeftModel


def get_last_hidden(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output[0]


def extract_state_dict_from_pth(pth_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    raise ValueError(f"Unrecognized checkpoint format: {pth_path}")


def normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        for pref in ["backbone.", "base_model.", "model.", "esm.", "bert.", "student."]:
            if k2.startswith(pref):
                k2 = k2[len(pref):]
        out[k2] = v
    return out


def ensure_int_list(x) -> List[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in list(x)]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [int(z) for z in list(v)]
            return [int(v)]
        except Exception:
            try:
                return [int(p) for p in s.split(",") if p.strip() != ""]
            except Exception:
                return []
    try:
        return [int(x)]
    except Exception:
        return []

class DNABERTEmbedderMutMean:
    def __init__(
        self,
        base_ckpt: str,
        length: int,
        device: str = "cuda:0",
        batch_size: int = 256,
        lora_adapter_dir: Optional[str] = None,
        ft_pth_path: Optional[str] = None,
        merge_lora: bool = False,
        max_tokens: int = 512,
        precision: str = 'fp32'
    ):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_tokens = int(max_tokens)
        self.precision = precision
        if self.precision == "bf16":
            self.target_dtype = torch.bfloat16
            self.use_amp_enabled = True
        elif self.precision == "fp16":
            self.target_dtype = torch.float16
            self.use_amp_enabled = True
        else:
            self.target_dtype = torch.float32
            self.use_amp_enabled = False

        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        config = BertConfig.from_pretrained(base_ckpt, num_labels=1)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_ckpt, 
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )

        if ft_pth_path and str(ft_pth_path).strip():
            sd_raw = extract_state_dict_from_pth(ft_pth_path)
            sd = normalize_state_dict_keys(sd_raw)
            missing, unexpected = base_model.load_state_dict(sd, strict=False)
            print(f"Loaded non-LoRA FT weights: {ft_pth_path}")
            print(f"   - missing keys   : {len(missing)}")
            print(f"   - unexpected keys: {len(unexpected)}")

        if lora_adapter_dir and str(lora_adapter_dir).strip():
            if os.path.isfile(lora_adapter_dir):
                raise ValueError(f"lora_adapter_dir must be a directory, got file: {lora_adapter_dir}")
            model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
            if merge_lora:
                model = model.merge_and_unload()
        else:
            model = base_model

        self.model = model.to(self.device).eval()
        
        if isinstance(self.model, PeftModel):
            self.backbone = self.model.get_base_model().bert
        else:
            self.backbone = self.model.bert
                    
        self.dim = int(getattr(self.backbone.config, "d_model", 
                              getattr(self.backbone.config, "hidden_size", 768)))
        print(f"Model loaded. Hidden dim: {self.dim}")

    @staticmethod
    def _repeat_embedding_vectors_fast(
        tokens: List[str], 
        embeddings: np.ndarray
    ) -> np.ndarray:
        token_lengths = np.array([len(t) for t in tokens])
        repeat_indices = np.repeat(np.arange(len(tokens)), token_lengths)
        return embeddings[:, repeat_indices, :]
    

    @torch.inference_mode()
    def _encode_to_per_base(self, seqs: List[str]) -> List[np.ndarray]:
        k = 6
        processed_seqs = []
        for seq in seqs:
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            processed_seqs.append(" ".join(kmers))

        encoded = self.tokenizer(
            processed_seqs, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens, 
            add_special_tokens=True, 
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.use_amp_enabled):
            outs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = get_last_hidden(outs).cpu().numpy()

        per_base_embs = []
        k = 6 
        
        for b in range(len(seqs)):
            seq_len = len(seqs[b])
            num_kmers = seq_len - k + 1
            

            valid_hidden = hidden[b, 1 : 1 + num_kmers, :]
            actual_n=valid_hidden.shape[0]
            
            base_res_emb = np.zeros((seq_len, self.dim), dtype=np.float32)
            
            base_res_emb[:actual_n, :] = valid_hidden
            

            if actual_n > 0:
                base_res_emb[actual_n:, :] = valid_hidden[-1, :]
                
            per_base_embs.append(base_res_emb)
                
        return per_base_embs

    def _forward_backbone(self, input_ids, attention_mask, L_tok):
        if L_tok > self.max_tokens:
            outs_chunks = []
            for start in range(0, L_tok, self.max_tokens):
                end = min(start + self.max_tokens, L_tok)
                out = self.backbone(
                    input_ids=input_ids[:, start:end],
                    attention_mask=attention_mask[:, start:end],
                )
                hidden = get_last_hidden(out)
                outs_chunks.append(hidden.detach().cpu())
            return torch.cat(outs_chunks, dim=1)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return get_last_hidden(out).detach().cpu()

    def get_mut_embeddings_from_df(
        self,
        seq_df: pd.DataFrame,
        seq_col: str,
        idx_col: str,
        out_col: str = "mut_emb_max",
    ) -> pd.DataFrame:
        
        if seq_col not in seq_df.columns:
            raise ValueError(f"Missing required column: {seq_col}")
        if idx_col not in seq_df.columns:
            raise ValueError(f"Missing required column: {idx_col}")

        out_rows = []
        n = len(seq_df)

        with torch.inference_mode():
            for start in tqdm(range(0, n, self.batch_size), 
                            desc="DNABERT-1 mutation pooling (max, per-base)"):
                end = min(start + self.batch_size, n)
                batch = seq_df.iloc[start:end]

                seqs = batch[seq_col].astype(str).tolist()
                mut_lists = batch[idx_col].tolist()

                per_base_embs = self._encode_to_per_base(seqs)  

                for i, (seq_emb, mut_idx_list) in enumerate(zip(per_base_embs, mut_lists)):
                    L_bp, H = seq_emb.shape
                    
                    mut_idx = ensure_int_list(mut_idx_list)
                    mut_idx = [j for j in mut_idx if 0 <= int(j) < L_bp]
                    
                    if len(mut_idx) == 0:
                        out_rows.append({out_col: np.full((self.dim,), np.nan, dtype=np.float32)})
                        continue

                    mut_idx = sorted(set(mut_idx))
                    
                    mut_embs = seq_emb[mut_idx, :]  
                    
                    pooled = mut_embs.max(axis=0)  
                    
                    out_rows.append({out_col: pooled.astype(np.float32)})

                del batch, seqs, mut_lists, per_base_embs
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return pd.DataFrame(out_rows)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", type=int, default=500)

    ap.add_argument("--variant_list_path", type=str, required=True)
    ap.add_argument("--dnv_path", type=str, required=True)

    ap.add_argument("--base_ckpt", type=str, 
                    default="zhihan1996/DNA_bert_6")

    ap.add_argument("--ft_pth_path", type=str, default="")
    ap.add_argument("--lora_adapter_dir", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=5000)
    ap.add_argument("--merge_lora", action="store_true")

    ap.add_argument("--max_tokens", type=int, default=512,
                    help="max tokens per forward pass (for long sequences)")


    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="Precision for inference (fp32, fp16, bf16)")

    ap.add_argument("--data_split", type=int, default=None,
                    help="Which split to process (0-indexed)")
    ap.add_argument("--num_splits", type=int, default=1,
                    help="Total number of splits")

    ap.add_argument("--out_path", type=str, required=True,
                    help="output feather path")
    
    ap.add_argument("--reverse_mode", type=str, default="forward",
                    help="(IGNORED for NT) kept for compatibility")
    ap.add_argument("--pooling", type=str, default="mean",
                    help="(IGNORED - always mean) kept for compatibility")

    return ap.parse_args()


def main():
    args = parse_args()

    bp = int(args.bp)
    seq_col = f"var_seq_{bp}bp"
    idx_col = f"mut_idx_{bp}bp"

    variant_list = pd.read_feather(args.variant_list_path)
    dnv = pd.read_feather(args.dnv_path)

    if seq_col not in dnv.columns or idx_col not in dnv.columns:
        raise ValueError(f"dnv must contain {seq_col} and {idx_col}")

    if args.data_split is not None and args.num_splits > 1:
        total_rows = len(dnv)
        split_size = total_rows // args.num_splits
        start_idx = args.data_split * split_size
        
        if args.data_split == args.num_splits - 1:
            end_idx = total_rows
        else:
            end_idx = start_idx + split_size
        
        print(f"[DATA SPLIT] Processing split {args.data_split}/{args.num_splits}")
        print(f"[DATA SPLIT] Rows: {start_idx} to {end_idx} (total: {total_rows})")
        
        variant_list = variant_list.iloc[start_idx:end_idx].reset_index(drop=True)
        dnv = dnv.iloc[start_idx:end_idx].reset_index(drop=True)

    max_len = int(dnv[seq_col].astype(str).str.len().max())
    print(f"Max sequence length: {max_len} bp")


    embedder = DNABERTEmbedderMutMean(
        base_ckpt=args.base_ckpt,
        length=max_len,
        device=args.device,
        batch_size=args.batch_size,
        lora_adapter_dir=args.lora_adapter_dir if args.lora_adapter_dir.strip() else None,
        ft_pth_path=args.ft_pth_path if args.ft_pth_path.strip() else None,
        merge_lora=args.merge_lora,
        max_tokens=args.max_tokens,
        precision=args.precision,
    )

    out_df = embedder.get_mut_embeddings_from_df(
        seq_df=dnv,
        seq_col=seq_col,
        idx_col=idx_col,
        out_col="mut_emb_max",
    )

    merged_df = pd.concat(
        [variant_list.reset_index(drop=True), out_df.reset_index(drop=True)],
        axis=1
    )

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged_df.to_feather(args.out_path)
    print("Saved:", args.out_path)
    print("Shape:", merged_df.shape)
    print("Columns:", list(merged_df.columns))


if __name__ == "__main__":
    main()