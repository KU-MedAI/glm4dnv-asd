import os
import re
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def safe_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".feather":
        return pd.read_feather(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file: {path}")


def safe_write_feather(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.reset_index(drop=True).to_feather(path)


def load_evo2(model_name: str):
    try:
        import evo2
        if hasattr(evo2, "Evo2"):
            return evo2.Evo2(model_name)
    except Exception:
        pass

    try:
        from evo2.models import Evo2
        return Evo2(model_name)
    except Exception as e:
        raise RuntimeError(f"Could not load evo2 model. err={e}")


class SimpleDNATokenizer:
    def __init__(self, pad_id: int = 0):
        self.pad_id = int(pad_id)
        self.map = {
            "A": 1, "C": 2, "G": 3, "T": 4, "N": 5,
            "a": 1, "c": 2, "g": 3, "t": 4, "n": 5,
        }

    def encode(self, seq: str, max_len: int) -> List[int]:
        ids = [self.map.get(ch, 5) for ch in str(seq)]
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def make_attention_mask(self, ids: List[int]) -> List[int]:
        return [0 if t == self.pad_id else 1 for t in ids]


def get_evo2_tokenizer(evo2_obj) -> Any:
    for attr in ["tokenizer", "tok", "encoder"]:
        if hasattr(evo2_obj, attr):
            return getattr(evo2_obj, attr)
    return SimpleDNATokenizer(pad_id=0)


def encode_batch(tokenizer, seqs: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if callable(tokenizer):
        try:
            out = tokenizer(
                seqs,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            input_ids = torch.tensor(out["input_ids"], dtype=torch.long)
            if "attention_mask" in out and out["attention_mask"] is not None:
                attention_mask = torch.tensor(out["attention_mask"], dtype=torch.long)
            else:
                attention_mask = (input_ids != 0).long()
            return input_ids, attention_mask
        except Exception:
            pass

    if hasattr(tokenizer, "encode"):
        ids = []
        masks = []
        fb = SimpleDNATokenizer(pad_id=0)

        for s in seqs:
            x = None
            try:
                if hasattr(tokenizer.encode, "__code__") and ("max_len" in tokenizer.encode.__code__.co_varnames):
                    x = tokenizer.encode(s, max_len=max_len)
                else:
                    x = tokenizer.encode(s)
            except Exception:
                x = None

            if x is None:
                x = fb.encode(s, max_len=max_len)

            x = list(x)
            if len(x) >= max_len:
                x = x[:max_len]
            else:
                x = x + [0] * (max_len - len(x))

            ids.append(x)

            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = 0
            pad_id = int(pad_id)
            masks.append([0 if t == pad_id else 1 for t in x])

        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    fb = SimpleDNATokenizer(pad_id=0)
    ids = [fb.encode(s, max_len=max_len) for s in seqs]
    masks = [fb.make_attention_mask(x) for x in ids]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / max(1, int(r))
        self.dropout = nn.Dropout(float(dropout))

        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype

        self.lora_A = nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r <= 0:
            return y
        if x.dtype != self.base.weight.dtype:
            x = x.to(dtype=self.base.weight.dtype)
        return y + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _named_modules_dict(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def apply_lora_to_linears_by_state_dict(backbone: nn.Module, sd_keys: List[str]) -> int:
    named = _named_modules_dict(backbone)
    targets = set()
    for k in sd_keys:
        if not k.startswith("backbone."):
            continue
        if ".lora_A." in k or ".lora_B." in k:
            mid = k[len("backbone."):]
            mid = mid.split(".lora_", 1)[0]
            targets.add(mid)

    wrapped = 0
    for module_path in sorted(targets):
        if module_path not in named:
            continue
        m = named[module_path]
        if not isinstance(m, nn.Linear):
            continue

        if "." in module_path:
            parent_path, child = module_path.rsplit(".", 1)
            parent = named.get(parent_path, None)
        else:
            parent = backbone
            child = module_path

        if parent is None or not hasattr(parent, child):
            continue

        base = getattr(parent, child)
        if not isinstance(base, nn.Linear):
            continue

        setattr(parent, child, LoRALinear(base, r=4, alpha=8, dropout=0.0))
        wrapped += 1

    return wrapped


def _rebuild_lora_modules_from_sd(backbone: nn.Module, sd: Dict[str, torch.Tensor], alpha_mult: int = 2) -> None:
    named = _named_modules_dict(backbone)

    groups: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        if ".lora_A.weight" in k:
            module_path = k[len("backbone."):].replace(".lora_A.weight", "")
            groups.setdefault(module_path, {})["A"] = v
        elif ".lora_B.weight" in k:
            module_path = k[len("backbone."):].replace(".lora_B.weight", "")
            groups.setdefault(module_path, {})["B"] = v

    for module_path, g in groups.items():
        if "A" not in g or "B" not in g:
            continue
        if module_path not in named:
            continue

        cur = named[module_path]
        if not isinstance(cur, LoRALinear):
            if isinstance(cur, nn.Linear):
                base_linear = cur
            else:
                continue
        else:
            base_linear = cur.base

        A = g["A"]
        B = g["B"]
        r = int(A.shape[0])
        in_f = int(A.shape[1])
        out_f = int(B.shape[0])

        if not (isinstance(base_linear, nn.Linear) and base_linear.in_features == in_f and base_linear.out_features == out_f):
            continue

        alpha = int(alpha_mult) * r

        if "." in module_path:
            parent_path, child = module_path.rsplit(".", 1)
            parent = named.get(parent_path, None)
        else:
            parent = backbone
            child = module_path
        if parent is None or not hasattr(parent, child):
            continue

        setattr(parent, child, LoRALinear(base_linear, r=r, alpha=alpha, dropout=0.0))


def load_ft_checkpoint_into_evo2(
    evo2_obj,
    ft_pth_path: str,
    device: torch.device,
    bf16: bool = True,
    lora_alpha_mult: int = 2,
) -> None:
    payload = torch.load(ft_pth_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        sd = payload["state_dict"]
    elif isinstance(payload, dict):
        sd = payload
    else:
        raise ValueError(f"Unsupported checkpoint format: type={type(payload)}")

    if not hasattr(evo2_obj, "model") or not isinstance(evo2_obj.model, nn.Module):
        raise RuntimeError("evo2_obj.model must exist and be nn.Module")

    backbone: nn.Module = evo2_obj.model
    sd_keys = list(sd.keys())
    has_lora = any(k.startswith("backbone.") and (".lora_A." in k or ".lora_B." in k) for k in sd_keys)

    if has_lora:
        apply_lora_to_linears_by_state_dict(backbone, sd_keys)
        _rebuild_lora_modules_from_sd(backbone, sd, alpha_mult=lora_alpha_mult)

        sd_backbone = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                sd_backbone[k[len("backbone."):]] = v

        backbone.load_state_dict(sd_backbone, strict=False)
    else:
        sd_backbone = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                sd_backbone[k[len("backbone."):]] = v
            else:
                sd_backbone[k] = v

        backbone.load_state_dict(sd_backbone, strict=False)

    backbone.to(device)
    if bf16:
        backbone.to(dtype=torch.bfloat16)
    backbone.eval()


def _maybe_get_embeddings_container(out: Any) -> Any:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            return out[1]
        if len(out) == 1:
            return out[0]
        return None
    if isinstance(out, dict):
        if "embeddings" in out:
            return out["embeddings"]
        return out
    return None


def _select_from_dict_by_layer(emb: Dict[str, Any], layer_name: str) -> torch.Tensor:
    if layer_name in emb and torch.is_tensor(emb[layer_name]):
        return emb[layer_name]

    tensor_vals = [(k, v) for k, v in emb.items() if torch.is_tensor(v)]
    if len(tensor_vals) == 1:
        return tensor_vals[0][1]

    cand = [(k, v) for k, v in tensor_vals if (layer_name in k or k in layer_name)]
    if len(cand) > 0:
        return cand[0][1]

    for _, v in emb.items():
        if isinstance(v, dict):
            try:
                return _select_from_dict_by_layer(v, layer_name)
            except Exception:
                pass

    if len(tensor_vals) > 0:
        return tensor_vals[0][1]

    raise RuntimeError(f"No tensor found in embeddings dict.")


def extract_hidden_for_layer(evo2_obj, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
    out = evo2_obj(input_ids, return_embeddings=True, layer_names=[layer_name])

    emb_container = _maybe_get_embeddings_container(out)
    if emb_container is None:
        raise RuntimeError(f"Could not get embeddings container.")

    if isinstance(emb_container, dict):
        if "embeddings" in emb_container and isinstance(emb_container["embeddings"], dict):
            emb_container = emb_container["embeddings"]
        hidden = _select_from_dict_by_layer(emb_container, layer_name)
        return hidden

    if isinstance(emb_container, (tuple, list)):
        for v in emb_container:
            if torch.is_tensor(v):
                return v
        raise RuntimeError("Embeddings container list/tuple had no tensor.")

    if torch.is_tensor(emb_container):
        return emb_container

    raise RuntimeError(f"Unsupported embeddings container type.")


def to_int_list(x) -> List[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in np.array(x).tolist()]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            import ast
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [int(t) for t in list(v)]
            return [int(v)]
        except Exception:
            try:
                return [int(p) for p in x.split(",") if p != ""]
            except Exception:
                return []
    try:
        return [int(x)]
    except Exception:
        return []


def mut_max_pool(hidden: torch.Tensor, mut_mask: torch.Tensor) -> torch.Tensor:
    if hidden.dim() != 3:
        raise RuntimeError(f"hidden must be (B,L,D).")
    if mut_mask.dim() != 2:
        raise RuntimeError(f"mut_mask must be (B,L).")

    neg_inf = torch.tensor(-1e9, device=hidden.device, dtype=hidden.dtype)
    masked = hidden.masked_fill(~mut_mask.unsqueeze(-1), neg_inf)
    pooled = masked.max(dim=1).values

    pooled = torch.where(
        torch.isclose(pooled, neg_inf),
        torch.tensor(float("nan"), device=hidden.device, dtype=hidden.dtype),
        pooled
    )
    return pooled


def maybe_reverse_seq(seq: str) -> str:
    return str(seq)[::-1]


def slice_by_split(df: pd.DataFrame, data_split: int, num_splits: int) -> pd.DataFrame:
    n = len(df)
    idx = np.arange(n)
    sel = idx[idx % num_splits == data_split]
    return df.iloc[sel].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--bp", type=int, required=True)
    ap.add_argument("--variant_list_path", type=str, required=True)
    ap.add_argument("--dnv_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--layer_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--reverse_mode", type=str, default="reverse", choices=["forward", "reverse"])
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--data_split", type=int, default=0)
    ap.add_argument("--num_splits", type=int, default=1)
    ap.add_argument("--verify_weights", type=int, default=0)
    ap.add_argument("--verify_n", type=int, default=2)
    ap.add_argument("--ft_pth_path", type=str, default="")
    ap.add_argument("--lora_alpha_mult", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=0)

    args = ap.parse_args()

    bp = int(args.bp)
    device = torch.device(args.device)
    bf16 = ("cuda" in str(device))

    seq_col = f"var_seq_{bp}bp"
    len_col = f"var_len_{bp}bp"
    idx_col = f"mut_idx_{bp}bp"

    dnv = safe_read_table(args.dnv_path)
    variant_list = safe_read_table(args.variant_list_path)

    vl_part = slice_by_split(variant_list, int(args.data_split), int(args.num_splits))
    uniq_vars = pd.unique(vl_part["variant"].astype(str))
    dnv_sub = dnv[dnv["variant"].astype(str).isin(uniq_vars)].copy()
    dnv_sub = dnv_sub.drop_duplicates(subset=["variant"], keep="first").reset_index(drop=True)

    var_seqs = dnv_sub[seq_col].astype(str).tolist()
    var_lens = dnv_sub[len_col].astype(int).tolist()
    mut_idxs = [to_int_list(x) for x in dnv_sub[idx_col].tolist()]

    if int(args.max_len) > 0:
        max_len = int(args.max_len)
    else:
        max_len = int(max(map(len, var_seqs))) if len(var_seqs) else 1
        max_len = max(1, max_len)

    evo2_obj = load_evo2(args.model_name)
    tokenizer = get_evo2_tokenizer(evo2_obj)

    if hasattr(evo2_obj, "model") and isinstance(evo2_obj.model, nn.Module):
        evo2_obj.model.to(device)
        if bf16:
            evo2_obj.model.to(dtype=torch.bfloat16)
        evo2_obj.model.eval()

    if str(args.ft_pth_path).strip():
        load_ft_checkpoint_into_evo2(
            evo2_obj,
            args.ft_pth_path,
            device=device,
            bf16=bf16,
            lora_alpha_mult=int(args.lora_alpha_mult),
        )

    all_mut: List[np.ndarray] = []
    bs = int(args.batch_size)

    it = range(0, len(var_seqs), bs)
    it = tqdm(it, total=(len(var_seqs) + bs - 1)//bs, leave=True)

    for s in it:
        chunk_seqs = var_seqs[s:s+bs]
        chunk_lens = var_lens[s:s+bs]
        chunk_midx = mut_idxs[s:s+bs]

        input_ids_fwd, _ = encode_batch(tokenizer, chunk_seqs, max_len=max_len)
        input_ids_fwd = input_ids_fwd.to(device)

        if args.reverse_mode == "reverse":
            rev_seqs = [maybe_reverse_seq(x) for x in chunk_seqs]
            input_ids_rev, _ = encode_batch(tokenizer, rev_seqs, max_len=max_len)
            input_ids_rev = input_ids_rev.to(device)
        else:
            input_ids_rev = None

        if bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h_fwd = extract_hidden_for_layer(evo2_obj, input_ids_fwd, args.layer_name)
                h_rev = extract_hidden_for_layer(evo2_obj, input_ids_rev, args.layer_name) if input_ids_rev is not None else None
        else:
            h_fwd = extract_hidden_for_layer(evo2_obj, input_ids_fwd, args.layer_name)
            h_rev = extract_hidden_for_layer(evo2_obj, input_ids_rev, args.layer_name) if input_ids_rev is not None else None

        B, L, D = h_fwd.shape
        mut_mask_fwd = torch.zeros((B, L), dtype=torch.bool, device=device)
        mut_mask_rev = torch.zeros((B, L), dtype=torch.bool, device=device) if input_ids_rev is not None else None

        for b in range(B):
            seq_len = int(chunk_lens[b])
            mids = [int(m) for m in chunk_midx[b] if 0 <= int(m) < seq_len and int(m) < L]
            if len(mids) == 0:
                continue
            mut_mask_fwd[b, mids] = True
            if mut_mask_rev is not None:
                rmids = [(seq_len - 1 - int(m)) for m in mids]
                rmids = [int(m) for m in rmids if 0 <= int(m) < L]
                if len(rmids) > 0:
                    mut_mask_rev[b, rmids] = True

        fwd_sel = mut_max_pool(h_fwd, mut_mask_fwd)
        if h_rev is not None:
            rev_sel = mut_max_pool(h_rev, mut_mask_rev)
            mut_concat = torch.cat([fwd_sel, rev_sel], dim=1)
        else:
            mut_concat = fwd_sel

        mut_np = mut_concat.detach().float().cpu().numpy().astype(np.float32)
        all_mut.append(mut_np)

    mut_emb = np.concatenate(all_mut, axis=0) if all_mut else np.zeros((0, 0), dtype=np.float32)

    var_emb_df = pd.DataFrame({
        "variant": dnv_sub["variant"].astype(str).values,
        "mut_emb": [x for x in mut_emb],
    })
    var_emb_df = var_emb_df.groupby("variant", as_index=False).agg({"mut_emb": "first"})

    vl_part2 = vl_part.copy()
    vl_part2["variant"] = vl_part2["variant"].astype(str)

    out_df = vl_part2.merge(
        var_emb_df,
        on="variant",
        how="left",
        validate="m:1"
    )
    out_df = out_df[["vcf_iid", "variant", "mut_emb"]]

    safe_write_feather(out_df, args.out_path)


if __name__ == "__main__":
    main()