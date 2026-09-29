"""Every matrix drawn in the video, computed with the repository's own FrameFT code.

The bases come from `peft.tuners.frame.layer.build_basis`, and the coefficient positions use the same
seeded permutation as `FrameLayer.update_layer` (entry_seed 2024, share_entry), so the pictures show
what the code actually builds. Results are cached in build/data.npz.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = HERE / "build" / "data.npz"
N, N_COEFFS, ENTRY_SEED = 768, 1000, 2024


def _positions(block):
    """Row/col indices of the 1,000 trainable entries, exactly as FrameLayer.update_layer picks them."""
    import torch
    per_block = block * block
    total = (N // block) * per_block
    idx = torch.randperm(total, generator=torch.Generator().manual_seed(ENTRY_SEED))[:N_COEFFS]
    sub, within = idx // per_block, idx % per_block
    rows = within // block + sub * block
    cols = within % block + sub * block
    return rows.numpy(), cols.numpy()


def _roberta_query():
    """Layer-0 query weight of roberta-base, if the weights were downloaded (see README); else None."""
    path = os.environ.get("ROBERTA_SAFETENSORS")
    if not path or not Path(path).exists():
        return None
    from safetensors import safe_open
    with safe_open(path, "np") as f:
        for key in f.keys():
            if key.endswith("encoder.layer.0.attention.self.query.weight"):
                return f.get_tensor(key).astype(np.float32)
    return None


def compute():
    sys.path.insert(0, str(REPO / "peft" / "src"))
    import torch
    from peft.tuners.frame.layer import build_basis

    out = {}
    for name in ("frame", "random", "identity"):
        out[f"B_{name}"] = build_basis(name, N, 2, 0).double().numpy()
    out["B_frame_l4"] = build_basis("frame", N, 4, 0).double().numpy()

    rng = np.random.default_rng(7)
    coeffs = rng.standard_normal(N_COEFFS)
    for block in (N, 2):
        r, c = _positions(block)
        S = np.zeros((N, N))
        S[r, c] = coeffs
        out[f"S_{block}"] = S
        out[f"rows_{block}"], out[f"cols_{block}"] = r, c
    S = out[f"S_{N}"]
    for name in ("frame", "random", "identity"):
        B = out[f"B_{name}"]
        out[f"dW_{name}"] = B.T @ S @ B
        out[f"sv_{name}"] = np.linalg.svd(out[f"dW_{name}"], compute_uv=False)
    out["dW_frame_block2"] = out["B_frame"].T @ out["S_2"] @ out["B_frame"]

    W = _roberta_query()
    if W is None:  # stand-in with a similar look: noise plus a few strong feature columns
        W = rng.standard_normal((N, N)).astype(np.float32) * 0.04
        W[:, [77, 588]] *= 6
    out["W"] = W
    np.savez_compressed(CACHE, **out)
    return out


def load():
    if CACHE.exists():
        return dict(np.load(CACHE))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    return compute()


if __name__ == "__main__":
    d = compute()
    for k, v in d.items():
        print(f"{k:18s} {v.shape} {v.dtype}")
    for name in ("frame", "random", "identity"):
        print(name, "top singular values", d[f"sv_{name}"][:3].round(4))
