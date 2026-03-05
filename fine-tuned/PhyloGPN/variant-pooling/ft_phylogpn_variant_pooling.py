#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import json
import argparse
import ast
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from safetensors import safe_open

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

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

def pad_sequence_phylogpn(seq: str, tokenizer, pad_size: int = 240) -> str:
    pad_token = tokenizer.pad_token
    return pad_token * pad_size + seq + pad_token * pad_size

def load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def load_lora_weights_manually(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0
        
    loaded_count = 0
    
    try:
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for checkpoint_key in f.keys():
                if 'lora_A' in checkpoint_key or 'lora_B' in checkpoint_key:
                    tensor = f.get_tensor(checkpoint_key)
                    
                    parts = checkpoint_key.split('.')
                    if 'phylogpn' in parts:
                        phylogpn_idx = parts.index('phylogpn')
                        parts[phylogpn_idx] = '_model'
                        
                        if parts[-1] == 'weight':
                            parts.insert(-1, 'default')
                        
                        model_key = '.'.join(parts)
                        
                        try:
                            param = model
                            for part in model_key.split('.'):
                                param = getattr(param, part)
                            param.data.copy_(tensor)
                            loaded_count += 1
                        except AttributeError:
                            continue
    except Exception as e:
        print(f"Manual loading error: {e}")
        return 0
    
    return loaded_count

def print_adapter_config_summary(config: Dict[str, Any]):
    if not config:
        print("No config loaded")
        return
    
    print("adapter_config.json summary:")
    fields = [
        ("peft_type", "peft_type"),
        ("task_type", "task_type"),
        ("r", "r"),
        ("lora_alpha", "lora_alpha"),
        ("lora_dropout", "lora_dropout"),
        ("target_modules", "target_modules"),
        ("base_model_name_or_path", "base_model"),
    ]
    
    for key, label in fields:
        if key in config:
            value = config[key]
            if isinstance(value, list):
                value_str = str(value)
            elif isinstance(value, str) and len(value) > 50:
                value_str = value.split("/")[-1]
            else:
                value_str = str(value)
            print(f"  - {label:<15}: {value_str}")

def detect_lora_modules_by_name(model) -> Dict[str, Any]:
    lora_a_names = []
    lora_b_names = []
    lora_params_total = 0
    
    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_a_names.append(name)
            lora_params_total += param.numel()
        elif "lora_B" in name:
            lora_b_names.append(name)
            lora_params_total += param.numel()
    
    return {
        "lora_A": len(lora_a_names),
        "lora_B": len(lora_b_names),
        "lora_A_names": lora_a_names,
        "lora_B_names": lora_b_names,
        "lora_params": lora_params_total
    }

def get_model_weight_sample(model, sample_size: int = 20) -> np.ndarray:
    for name, param in model.named_parameters():
        flat = param.detach().cpu().flatten()
        if len(flat) >= sample_size:
            return flat[:sample_size].numpy()
    
    for param in model.parameters():
        flat = param.detach().cpu().flatten()
        if len(flat) >= sample_size:
            return flat[:sample_size].numpy()
    
    return np.zeros(sample_size)

def generate_test_sequence(length: int = 100) -> str:
    bases = "ACGT"
    return "".join([bases[i % 4] for i in range(length)])

def test_model_forward(model, tokenizer, device: torch.device, seq_length: int = 100, pad_size: int = 240):
    try:
        model = model.to(device)
        test_seq = generate_test_sequence(seq_length)
        test_seq_padded = pad_sequence_phylogpn(test_seq, tokenizer, pad_size)
        
        encoded = tokenizer(
            [test_seq_padded], 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=seq_length + 2*pad_size, 
            add_special_tokens=False,
        )
        
        input_ids = encoded["input_ids"].to(device)
        
        with torch.inference_mode():
            if isinstance(model, PeftModel):
                rcebytenet = model.base_model.model.phylogpn
            else:
                rcebytenet = model.phylogpn
            
            x = rcebytenet.embedding(input_ids)
            x = x.transpose(1, 2)
            x = rcebytenet.blocks(x)
            x = rcebytenet.output_layers[0](x)
            x = x.transpose(1, 2)
            
            return x.mean(dim=1).cpu()
            
    except Exception as e:
        print(f"Forward test error: {e}")
        return None

def compare_outputs(output_pre, output_post) -> Dict[str, float]:
    if output_pre is None or output_post is None:
        return {}
    
    pre = output_pre.flatten().numpy()
    post = output_post.flatten().numpy()
    
    abs_diff = np.abs(pre - post)
    l2_diff = np.linalg.norm(pre - post)
    
    dot = np.dot(pre, post)
    norm_pre = np.linalg.norm(pre)
    norm_post = np.linalg.norm(post)
    cos_sim = dot / (norm_pre * norm_post + 1e-8)
    
    return {
        "max_abs_diff": abs_diff.max(),
        "mean_abs_diff": abs_diff.mean(),
        "l2_diff": l2_diff,
        "mean_l2_diff": l2_diff / len(pre),
        "cos_sim": cos_sim
    }

def verify_lora_impact(model):
    try:
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A = module.lora_A['default'].weight
                lora_B = module.lora_B['default'].weight
                scaling = getattr(module, 'scaling', {}).get('default', 1.0)
                base_weight = module.base_layer.weight
                
                lora_A_2d = lora_A.squeeze(-1) if lora_A.dim() > 2 else lora_A
                lora_B_2d = lora_B.squeeze(-1) if lora_B.dim() > 2 else lora_B
                
                lora_delta = (lora_B_2d @ lora_A_2d) * scaling
                
                base_norm = base_weight.norm().item()
                delta_norm = lora_delta.norm().item()
                relative_impact = (delta_norm / base_norm) * 100
                
                print(f"LORA IMPACT: {name}")
                print(f"  - Base weight norm: {base_norm:.6f}")
                print(f"  - LoRA delta norm: {delta_norm:.6f}")
                print(f"  - Relative impact: {relative_impact:.3f}%")
                
                if relative_impact > 0.1:
                    print(f"  Meaningful LoRA impact detected!")
                else:
                    print(f"  Low LoRA impact")
                
                return True
                
    except Exception as e:
        print(f"Verification error: {e}")
        return False
    
    return False

class PhyloGPNForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels=1):
        super().__init__()
        self.phylogpn = base_model._model
        self.num_labels = num_labels
        self.config = base_model.config
        embedding_dim = base_model.config.outer_dim
        self.classifier = nn.Linear(embedding_dim, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.phylogpn.embedding(input_ids)
        x = x.transpose(1, 2)
        x = self.phylogpn.blocks(x)
        x = self.phylogpn.output_layers[0](x)
        x = x.transpose(1, 2)
        pooled_output = x.mean(dim=1)
        logits = self.classifier(pooled_output)
        
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=logits)

class PhyloGPNEmbedderMutMax:
    def __init__(
        self,
        base_ckpt: str = "songlab/PhyloGPN",
        lora_adapter_dir: Optional[str] = None,
        revision: Optional[str] = None,
        length: int = 2200,
        device: str = "cuda:0",
        batch_size: int = 128,
        merge_lora: bool = False,
        use_amp: bool = True,
    ):
        print("=" * 60)
        print("Starting PhyloGPN Model Initialization")
        print("=" * 60)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.length = int(length)
        self.pad_flank = 240
        
        print(f"Loading from {base_ckpt}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_ckpt, revision=revision, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Tokenizer loaded (pad_token: '{self.tokenizer.pad_token}')")

        print(f"Loading from {base_ckpt}")
        base_model_raw = AutoModel.from_pretrained(
            base_ckpt, revision=revision, trust_remote_code=True
        )
        base_model = PhyloGPNForSequenceClassification(base_model_raw, num_labels=1)
        print(f"Base model with wrapper loaded")

        current_model = base_model
        
        if lora_adapter_dir and str(lora_adapter_dir).strip():
            print("=" * 80)
            print("Starting Complete LoRA Loading & Verification")
            print("=" * 80)
            print(f"loading LoRA adapter: {lora_adapter_dir}")
            
            if not os.path.isdir(lora_adapter_dir):
                raise FileNotFoundError(f"Adapter dir not found: {lora_adapter_dir}")
            
            print(f"Loading adapter_config.json...")
            adapter_config = load_adapter_config(lora_adapter_dir)
            print_adapter_config_summary(adapter_config)
            
            print(f"Loading adapter...")
            try:
                current_model = PeftModel.from_pretrained(base_model, lora_adapter_dir, is_trainable=False)
                print(f"Loaded successfully (no key mapping needed)")
            except Exception as e:
                print(f"Failed: {e}")
                raise

            print(f"Detecting modules...")
            lora_info = detect_lora_modules_by_name(current_model)
            print(f"detected lora modules by param name: lora_A={lora_info['lora_A']} | lora_B={lora_info['lora_B']}")
            
            print(f"Testing LoRA-loaded model...")
            output_post = test_model_forward(current_model, self.tokenizer, self.device, 100, self.pad_flank)
            if output_post is not None:
                print(f"LoRA model forward OK")

            print(f"Analyzing LoRA impact...")
            verify_lora_impact(current_model)
            
            if lora_info['lora_params'] == 0:
                raise RuntimeError("LoRA params = 0! Loading FAILED!")
            
            if merge_lora:
                print(f"Merging weights...")
                current_model = current_model.merge_and_unload()
                print(f"Merged")
            
            print(f"device={self.device} | length={self.length} | LoRA mode: {'merged' if merge_lora else 'adapter loaded (not merged)'}")
            print("=" * 80)
        else:
            print("No adapter specified")

        self.model = current_model.to(self.device).eval()
        print(f"Model moved to {self.device}")
        
        self.dim = self._get_embedding_dim()
        
        print("=" * 60)
        print("PhyloGPN Model Initialization Complete!")
        print(f"  - Model type: {type(self.model).__name__}")
        print(f"  - Hidden dimension: {self.dim}")
        print(f"  - Device: {self.device}")
        print(f"  - AMP enabled: {self.use_amp}")
        print("=" * 60)

    def _get_embedding_dim(self):
        try:
            if isinstance(self.model, PeftModel):
                config = self.model.base_model.config
            else:
                config = getattr(self.model, 'config', None)
                
            dim = getattr(config, "outer_dim", None) or getattr(config, "hidden_size", None)
            
            if dim is None:
                dim = 960
                
        except Exception as e:
            dim = 960
        
        return dim

    def _get_phylogpn_model(self):
        if isinstance(self.model, PeftModel):
            return self.model.base_model.model.phylogpn
        else:
            return self.model.phylogpn

    def _get_embeddings(self, input_ids):
        rcebytenet = self._get_phylogpn_model()
        
        try:
            x = rcebytenet.embedding(input_ids)
            x = x.transpose(1, 2)
            x = rcebytenet.blocks(x)
            x = rcebytenet.output_layers[0](x)
            x = x.transpose(1, 2)
            return x
        except Exception as e:
            raise RuntimeError(f"Failed to extract embeddings: {e}")

    @torch.inference_mode()
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

        self.model.eval()

        for start in tqdm(range(0, n, self.batch_size), 
                        desc="PhyloGPN mutation pooling (max)"):
            end = min(start + self.batch_size, n)
            batch = seq_df.iloc[start:end]

            seqs_raw = batch[seq_col].astype(str).tolist()
            mut_lists = batch[idx_col].tolist()

            seqs_padded = [pad_sequence_phylogpn(s, self.tokenizer, self.pad_flank) for s in seqs_raw]

            toks = self.tokenizer(
                seqs_padded,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=self.length,
                return_tensors="pt",
            )
            input_ids = toks["input_ids"].to(self.device)

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    token_embs = self._get_embeddings(input_ids)
            else:
                token_embs = self._get_embeddings(input_ids)
            
            token_embs = token_embs.detach().cpu().numpy()
            B, L_emb, D = token_embs.shape

            for b in range(B):
                seq_raw = seqs_raw[b]
                mut_idx_list = mut_lists[b]
                seq_len = len(seq_raw)

                mut_idx = ensure_int_list(mut_idx_list)
                mut_idx = [j for j in mut_idx if 0 <= int(j) < seq_len]
                
                if len(mut_idx) == 0:
                    out_rows.append({out_col: np.full((self.dim,), np.nan, dtype=np.float32)})
                    continue

                mut_idx = sorted(set(mut_idx))
                
                adjusted_indices = [mi + self.pad_flank for mi in mut_idx]
                valid_indices = [idx for idx in adjusted_indices if 0 <= idx < L_emb]
                
                if len(valid_indices) == 0:
                    out_rows.append({out_col: np.full((self.dim,), np.nan, dtype=np.float32)})
                    continue

                mut_embs = token_embs[b, valid_indices, :]
                pooled = mut_embs.max(axis=0)
                out_rows.append({out_col: pooled.astype(np.float32)})

            del toks, input_ids, token_embs
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return pd.DataFrame(out_rows)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", type=int, default=500)
    ap.add_argument("--variant_list_path", type=str, required=True)
    ap.add_argument("--dnv_path", type=str, required=True)
    ap.add_argument("--base_ckpt", type=str, default="songlab/PhyloGPN")
    ap.add_argument("--lora_adapter_dir", type=str, default="")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=5000)
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--data_split", type=int, default=None)
    ap.add_argument("--num_splits", type=int, default=1)
    ap.add_argument("--out_path", type=str, required=True)
    
    ap.add_argument("--ft_pth_path", type=str, default="")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--reverse_mode", type=str, default="forward")
    ap.add_argument("--pooling", type=str, default="mean")

    return ap.parse_args()

def main():
    args = parse_args()

    bp = int(args.bp)
    seq_col = f"ref_seq_{bp}bp"
    idx_col = f"ref_idx_{bp}bp"

    print("Loading variant_list:", args.variant_list_path)
    variant_list = pd.read_feather(args.variant_list_path)

    print("Loading dnv         :", args.dnv_path)
    dnv = pd.read_feather(args.dnv_path)

    if seq_col not in dnv.columns or idx_col not in dnv.columns:
        raise ValueError(f"dnv must contain {seq_col} and {idx_col}")

    if args.data_split is not None and args.num_splits > 1:
        ds, ns = int(args.data_split), int(args.num_splits)
        if ds < 0 or ds >= ns:
            raise ValueError(f"--data_split must be in [0, {ns-1}]. Got {ds}")

        total_rows = len(dnv)
        split_size = total_rows // ns
        start_idx = ds * split_size
        end_idx = total_rows if ds == ns - 1 else start_idx + split_size
        
        print(f"split {ds}/{ns} | rows {start_idx}:{end_idx} (total={total_rows})")
        
        variant_list = variant_list.iloc[start_idx:end_idx].reset_index(drop=True)
        dnv = dnv.iloc[start_idx:end_idx].reset_index(drop=True)

    max_len_raw = int(dnv[seq_col].astype(str).str.len().max())
    max_len = max_len_raw + 2 * 240
    print(f"bp={bp} | seq_col={seq_col} | idx_col={idx_col} | max_len={max_len_raw}+480={max_len}")

    embedder = PhyloGPNEmbedderMutMax(
        base_ckpt=args.base_ckpt,
        lora_adapter_dir=args.lora_adapter_dir if args.lora_adapter_dir.strip() else None,
        revision=args.revision,
        length=max_len,
        device=args.device,
        batch_size=args.batch_size,
        merge_lora=args.merge_lora,
        use_amp=args.use_amp,
    )

    out_df = embedder.get_mut_embeddings_from_df(
        seq_df=dnv,
        seq_col=seq_col,
        idx_col=idx_col,
        out_col="mut_emb_max",
    )

    merged_df = pd.concat([variant_list.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged_df.to_feather(args.out_path)

    emb_dim = None
    try:
        emb_dim = int(len(merged_df["mut_emb_max"].iloc[0]))
    except Exception:
        pass

    print("saved:", args.out_path)
    print("shape:", merged_df.shape)
    if emb_dim is not None:
        print("embedding dim:", emb_dim)

if __name__ == "__main__":
    main()