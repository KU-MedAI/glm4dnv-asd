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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


# ============================================================
# Utils
# ============================================================
def get_last_hidden(output) -> torch.Tensor:
    """Extract last hidden state from model output"""
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output[0]


def extract_state_dict_from_pth(pth_path: str) -> Dict[str, torch.Tensor]:
    """Extract state dict from .pth checkpoint"""
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
    """Remove common prefixes from state dict keys"""
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        for pref in ["backbone.", "base_model.", "model.", "esm."]:
            if k2.startswith(pref):
                k2 = k2[len(pref):]
        out[k2] = v
    return out


def ensure_int_list(x) -> List[int]:
    """
    Convert mut_idx column to [int, int, ...] list
    Handles various input formats: list/tuple/ndarray/scalar/string
    """
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


# ============================================================
# LoRA Verification Helper Functions
# ============================================================
def load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    """Load and parse adapter_config.json"""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def print_adapter_config_summary(config: Dict[str, Any]):
    """Print formatted adapter config summary"""
    if not config:
        print("[LORA] No config loaded")
        return
    
    print("[LORA] adapter_config.json summary:")
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
                value_str = value.split("/")[-1]  # Show only model name
            else:
                value_str = str(value)
            print(f"  - {label:<15}: {value_str}")


def detect_lora_modules_by_name(model) -> Dict[str, Any]:
    """Detect LoRA modules by parameter names"""
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
    """Extract sample of model weights"""
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
    """Generate test DNA sequence"""
    bases = "ACGT"
    return "".join([bases[i % 4] for i in range(length)])


def test_model_forward(model, tokenizer, device: torch.device, seq_length: int = 100):
    """Test forward pass with dummy sequence"""
    try:
        model = model.to(device) 
        
        test_seq = generate_test_sequence(seq_length)
        
        encoded = tokenizer(
            [test_seq], return_tensors="pt", padding=True,
            truncation=True, max_length=seq_length, add_special_tokens=False,
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.inference_mode():
            if isinstance(model, PeftModel):
                backbone = model.get_base_model().esm
            else:
                backbone = model.esm if hasattr(model, 'esm') else model
            
            output = backbone(input_ids=input_ids, attention_mask=attention_mask)
            
            if isinstance(output, tuple):
                hidden = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            else:
                hidden = output
            
            return hidden.mean(dim=1).cpu()
            
    except Exception as e:
        print(f"[VERIFY] Forward test error: {e}")
        return None


def compare_outputs(output_pre, output_post) -> Dict[str, float]:
    """Compare model outputs"""
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


# ============================================================
# Weight Loading Verification Functions
# ============================================================
def get_model_weight_signature(model, layer_name: str = "word_embeddings") -> np.ndarray:
    """
    Extract a small signature from model weights for verification.
    This helps confirm that weights have actually changed after loading.
    """
    try:
        if hasattr(model, 'esm') and hasattr(model.esm, 'embeddings'):
            if layer_name == "word_embeddings" and hasattr(model.esm.embeddings, 'word_embeddings'):
                weight = model.esm.embeddings.word_embeddings.weight
                return weight[0:3, 0:5].detach().cpu().numpy().flatten()
        
        # Fallback: get first available parameter
        for param in model.parameters():
            if param.numel() >= 15:  # At least 15 elements
                return param.flatten()[:15].detach().cpu().numpy()
        
        return np.array([0.0])  # Ultimate fallback
    except Exception as e:
        print(f"[WEIGHT CHECK] Error getting signature: {e}")
        return np.array([0.0])


def verify_model_weights_changed(before_sig: np.ndarray, after_sig: np.ndarray, 
                                operation: str = "loading") -> bool:
    """Check if model weights have actually changed"""
    try:
        if len(before_sig) != len(after_sig):
            print(f"[WEIGHT CHECK] {operation}: Signature length changed ({len(before_sig)} → {len(after_sig)})")
            return True
            
        diff = np.abs(before_sig - after_sig).sum()
        changed = diff > 1e-6
        
        if changed:
            print(f"[WEIGHT CHECK] {operation}: Weights CHANGED (diff={diff:.6f})")
            print(f"   Before: {before_sig[:3]} ...")
            print(f"   After:  {after_sig[:3]} ...")
        else:
            print(f"[WEIGHT CHECK] {operation}: Weights UNCHANGED (diff={diff:.6f})")
            
        return changed
    except Exception as e:
        print(f"[WEIGHT CHECK] Error comparing signatures: {e}")
        return False


def check_lora_parameters(model) -> Dict[str, int]:
    """Check LoRA parameter counts"""
    if not isinstance(model, PeftModel):
        return {"total_params": sum(p.numel() for p in model.parameters())}
    
    base_params = sum(p.numel() for p in model.get_base_model().parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "base_params": base_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_params": total_params - base_params
    }


# ============================================================
# NT Mutation Pooling Embedder (Per-base Upsampling)
# ============================================================
class NTEmbedderMutMax:
    """
    Nucleotide Transformer mutation pooling with per-base upsampling:
    - Load fine-tuned NT model (AutoModelForSequenceClassification)
    - Upsample k-mer token embeddings to per-base embeddings
    - Extract embeddings only at mutation positions
    - Apply MAX pooling over mutation tokens -> (D,)
    - Output column: mut_emb_max (numpy array shape (D,))
    """

    def __init__(
        self,
        base_ckpt: str,
        length: int,
        device: str = "cuda:0",
        batch_size: int = 256,
        lora_adapter_dir: Optional[str] = None,
        ft_pth_path: Optional[str] = None,
        merge_lora: bool = False,
        max_tokens: int = 2048,
        use_amp: bool = True,
        verify_weights: bool = True,
    ):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_tokens = int(max_tokens)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.verify_weights = verify_weights

        print("=" * 60)
        print("[MODEL LOADING] Starting NT Model Initialization")
        print("=" * 60)

        # Tokenizer
        print(f"[TOKENIZER] Loading from {base_ckpt}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Base model loading
        print(f"[BASE MODEL] Loading from {base_ckpt}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_ckpt, 
            num_labels=1,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        # Get baseline weight signature
        if self.verify_weights:
            base_signature = get_model_weight_signature(base_model)
            print(f"[WEIGHT CHECK] Base model signature: {base_signature[:3]}")

        # Load non-LoRA checkpoint if provided
        if ft_pth_path and str(ft_pth_path).strip():
            print(f"[FT WEIGHTS] Loading fine-tuned weights: {ft_pth_path}")
            if not os.path.exists(ft_pth_path):
                raise FileNotFoundError(f"Fine-tuned weights not found: {ft_pth_path}")
                
            sd_raw = extract_state_dict_from_pth(ft_pth_path)
            sd = normalize_state_dict_keys(sd_raw)
            missing, unexpected = base_model.load_state_dict(sd, strict=False)
            
            print(f"[FT WEIGHTS] Loaded successfully!")
            print(f"   - Total keys in checkpoint: {len(sd)}")
            print(f"   - Missing keys: {len(missing)}")
            print(f"   - Unexpected keys: {len(unexpected)}")
            
            if self.verify_weights:
                ft_signature = get_model_weight_signature(base_model)
                verify_model_weights_changed(base_signature, ft_signature, "fine-tuned weights")
                base_signature = ft_signature

        # Apply LoRA adapter with complete verification
        current_model = base_model
        
        if lora_adapter_dir and str(lora_adapter_dir).strip():
            print("=" * 80)
            print("[LORA] Starting Complete LoRA Loading & Verification")
            print("=" * 80)
            print(f"[LORA] loading LoRA adapter: {lora_adapter_dir}")
            
            if not os.path.isdir(lora_adapter_dir):
                raise FileNotFoundError(f"Adapter dir not found: {lora_adapter_dir}")
            
            # Load config
            print(f"\n[LORA] Loading adapter_config.json...")
            adapter_config = load_adapter_config(lora_adapter_dir)
            print_adapter_config_summary(adapter_config)
            
            # PRE-LoRA test
            print(f"\n[VERIFY] pre-LoRA test...")
            if self.verify_weights:
                weight_pre = get_model_weight_sample(base_model)
            
            output_pre = test_model_forward(base_model, self.tokenizer, self.device, 100)
            if output_pre is not None:
                print(f"[VERIFY] pre-LoRA forward OK")
            
            # Load LoRA
            print(f"\n[LORA] Loading adapter...")
            try:
                current_model = PeftModel.from_pretrained(base_model, lora_adapter_dir, is_trainable=False)
                print(f"[LORA] Loaded successfully")
            except Exception as e:
                print(f"❌ [LORA] Failed: {e}")
                raise
            
            # Detect modules
            print(f"\n[LORA] Detecting modules...")
            lora_info = detect_lora_modules_by_name(current_model)
            print(f"[LORA] detected lora modules by param name: lora_A={lora_info['lora_A']} | lora_B={lora_info['lora_B']}")
            
            # POST-LoRA test
            print(f"\n[VERIFY] post-LoRA test...")
            output_post = test_model_forward(current_model, self.tokenizer, self.device, 100)
            
            # Compare
            if output_pre is not None and output_post is not None:
                comp = compare_outputs(output_pre, output_post)
                print(f"[VERIFY] PRE_LORA vs AFTER_LORA: max_abs_diff={comp['max_abs_diff']:.5f} | mean_L2_diff={comp['mean_l2_diff']:.5f} | mean_cos={comp['cos_sim']:.6f}")
                
                if comp['max_abs_diff'] < 1e-6:
                    print(f"[WARNING] Outputs identical! LoRA may not be active!")
            
            if lora_info['lora_params'] == 0:
                raise RuntimeError("LoRA params = 0! Loading FAILED!")
            
            # Merge
            if merge_lora:
                print(f"\n[LORA] Merging weights...")
                current_model = current_model.merge_and_unload()
                print(f"[LORA] Merged")
            
            print(f"\n[MODEL] device={self.device} | length={self.length} | LoRA mode: {'merged' if merge_lora else 'adapter loaded (not merged)'}")
            print("=" * 80)
        else:
            print("[LORA] No adapter specified")

        # Move to device and set eval mode
        self.model = current_model.to(self.device).eval()
        print(f"[DEVICE] Model moved to {self.device}")
        
        # Extract ESM backbone and hidden dimension
        if isinstance(self.model, PeftModel):
            self.backbone = self.model.get_base_model().esm
        else:
            self.backbone = self.model.esm
            
        self.dim = int(getattr(self.backbone.config, "d_model", 
                              getattr(self.backbone.config, "hidden_size", 256)))
        
        # Final model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("[MODEL READY] NT Model Initialization Complete!")
        print(f"   - Model type: {type(self.model).__name__}")
        print(f"   - Hidden dimension: {self.dim}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Device: {self.device}")
        print(f"   - AMP enabled: {self.use_amp}")
        print("=" * 60)


    @staticmethod
    def _repeat_embedding_vectors_fast(
        tokens: List[str], 
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Fast vectorized upsampling"""

        token_lengths = np.array([len(t) for t in tokens])
        repeat_indices = np.repeat(np.arange(len(tokens)), token_lengths)

        return embeddings[:, repeat_indices, :]

    @torch.inference_mode()
    def _encode_to_per_base(self, seqs: List[str]) -> List[np.ndarray]:
        encoded = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)  # 사용
        B, L_tok = input_ids.shape

        # Forward pass
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outs = self._forward_backbone(input_ids, attention_mask, L_tok)
        else:
            outs = self._forward_backbone(input_ids, attention_mask, L_tok)
        
        outs_np = outs.numpy()
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = attention_mask.cpu().numpy()  # 추가
        
        per_base_embs = []
        for b in range(B):
            # Valid tokens only (attention_mask == 1)
            valid_mask = attention_mask_cpu[b] == 1
            valid_token_indices = np.where(valid_mask)[0]
            
            tokens_b = self.tokenizer.convert_ids_to_tokens(input_ids_cpu[b])
            valid_tokens = [tokens_b[i] for i in valid_token_indices]
            
            emb_b = outs_np[b:b+1, valid_token_indices, :]  # Valid embeddings only
            
            # Upsampling
            emb_b = self._repeat_embedding_vectors_fast(valid_tokens, emb_b)
            per_base_embs.append(emb_b[0])
        
        return per_base_embs

    def _forward_backbone(self, input_ids, attention_mask, L_tok):
        """Separated forward logic"""
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
        """
        Extract mutation embeddings from dataframe.
        
        Args:
            seq_df: DataFrame with sequences and mutation indices
            seq_col: column name for sequences
            idx_col: column name for mutation indices
            out_col: output column name for embeddings
            
        Returns:
            DataFrame with mutation embeddings
        """
        if seq_col not in seq_df.columns:
            raise ValueError(f"Missing required column: {seq_col}")
        if idx_col not in seq_df.columns:
            raise ValueError(f"Missing required column: {idx_col}")

        out_rows = []
        n = len(seq_df)

        with torch.inference_mode():
            for start in tqdm(range(0, n, self.batch_size), 
                            desc="NT mutation pooling (max, per-base)"):
                end = min(start + self.batch_size, n)
                batch = seq_df.iloc[start:end]

                seqs = batch[seq_col].astype(str).tolist()
                mut_lists = batch[idx_col].tolist()

                # Get per-base embeddings for all sequences in batch
                per_base_embs = self._encode_to_per_base(seqs)

                # Extract mutation positions for each sequence
                for i, (seq_emb, mut_idx_list) in enumerate(zip(per_base_embs, mut_lists)):
                    L_bp, H = seq_emb.shape
                    
                    # Get valid mutation indices
                    mut_idx = ensure_int_list(mut_idx_list)
                    mut_idx = [j for j in mut_idx if 0 <= int(j) < L_bp]
                    
                    if len(mut_idx) == 0:
                        out_rows.append({out_col: np.full((self.dim,), np.nan, dtype=np.float32)})
                        continue

                    mut_idx = sorted(set(mut_idx))
                    
                    mut_embs = seq_emb[mut_idx, :]  # (K, H) where K = num mutations
                    
                    pooled = mut_embs.max(axis=0)  # (H,)
                    
                    out_rows.append({out_col: pooled.astype(np.float32)})

                # Cleanup
                del batch, seqs, mut_lists, per_base_embs
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return pd.DataFrame(out_rows)


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", type=int, default=500)

    ap.add_argument("--variant_list_path", type=str, required=True)
    ap.add_argument("--dnv_path", type=str, required=True)

    ap.add_argument("--base_ckpt", type=str, 
                    default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")

    ap.add_argument("--ft_pth_path", type=str, default="")
    ap.add_argument("--lora_adapter_dir", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=5000)
    ap.add_argument("--merge_lora", action="store_true")

    ap.add_argument("--max_tokens", type=int, default=2048,
                    help="max tokens per forward pass (for long sequences)")
    
    ap.add_argument("--use_amp", action="store_true",
                    help="use automatic mixed precision (fp16)")
    
    ap.add_argument("--skip_weight_verification", action="store_true",
                    help="skip weight loading verification (faster startup)")

    ap.add_argument("--data_split", type=int, default=None,
                    help="Which split to process (0-indexed)")
    ap.add_argument("--num_splits", type=int, default=1,
                    help="Total number of splits")

    ap.add_argument("--out_path", type=str, required=True,
                    help="output feather path")
    
    ap.add_argument("--reverse_mode", type=str, default="forward",
                    help="(IGNORED for NT) kept for compatibility")
    ap.add_argument("--pooling", type=str, default="max",
                    help="(IGNORED - always max) kept for compatibility")

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    bp = int(args.bp)
    seq_col = f"var_seq_{bp}bp"
    idx_col = f"mut_idx_{bp}bp"

    print("[INFO] Loading variant_list:", args.variant_list_path)
    variant_list = pd.read_feather(args.variant_list_path)

    print("[INFO] Loading dnv        :", args.dnv_path)
    dnv = pd.read_feather(args.dnv_path)

    if seq_col not in dnv.columns or idx_col not in dnv.columns:
        raise ValueError(f"dnv must contain {seq_col} and {idx_col}")

    # Row-wise split
    if args.data_split is not None and int(args.num_splits) > 1:
        ds = int(args.data_split)
        ns = int(args.num_splits)
        if ds < 0 or ds >= ns:
            raise ValueError(f"--data_split must be in [0, num_splits-1]. Got data_split={ds}, num_splits={ns}")

        total_rows = len(dnv)
        if len(variant_list) != total_rows:
            raise ValueError(
                f"variant_list and dnv must have same #rows for split. "
                f"Got len(variant_list)={len(variant_list)} vs len(dnv)={len(dnv)}"
            )

        split_size = total_rows // ns
        start_idx = ds * split_size
        end_idx = total_rows if (ds == ns - 1) else (start_idx + split_size)

        print(f"[DATA SPLIT] split {ds}/{ns} | rows {start_idx}:{end_idx} (total={total_rows})")

        variant_list = variant_list.iloc[start_idx:end_idx].reset_index(drop=True)
        dnv = dnv.iloc[start_idx:end_idx].reset_index(drop=True)

    max_len = int(dnv[seq_col].astype(str).str.len().max())
    print(f"[INFO] bp={bp} | seq_col={seq_col} | idx_col={idx_col} | max_len={max_len}")

    # Initialize embedder
    embedder = NTEmbedderMutMax(
        base_ckpt=args.base_ckpt,
        length=max_len,
        device=args.device,
        batch_size=args.batch_size,
        lora_adapter_dir=args.lora_adapter_dir if args.lora_adapter_dir.strip() else None,
        ft_pth_path=args.ft_pth_path if args.ft_pth_path.strip() else None,
        merge_lora=args.merge_lora,
        max_tokens=args.max_tokens,
        use_amp=args.use_amp,
        verify_weights=not args.skip_weight_verification,
    )

    # Extract mutation embeddings
    out_df = embedder.get_mut_embeddings_from_df(
        seq_df=dnv,
        seq_col=seq_col,
        idx_col=idx_col,
        out_col="mut_emb_max",
    )

    # Merge with variant list
    merged_df = pd.concat(
        [variant_list.reset_index(drop=True), out_df.reset_index(drop=True)],
        axis=1
    )

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
    print("   shape:", merged_df.shape)
    if emb_dim is not None:
        print("   embedding dim:", emb_dim)


if __name__ == "__main__":
    main()