"""Sanity checks for the --basis switch. Runs on CPU in a few seconds: `python check_bases.py`."""
import math

import torch
import torch.nn as nn

from peft.tuners.frame import layer as frame_layer
from peft.tuners.frame.layer import Linear, build_basis, construct_real_tff

n = 768
eye = torch.eye(n, dtype=torch.float64)

# 1. every basis is orthogonal (the frame is built in float32, hence the looser tolerance)
for basis in ("frame", "random", "identity"):
    for l in (2, 4):
        B = build_basis(basis, n, l, basis_seed=0).double()
        err = torch.linalg.norm(B @ B.T - eye).item()
        assert err < 1e-2, f"{basis} l={l} not orthogonal: {err}"
        print(f"ok  {basis:8s} l={l}: orthogonal (||BB^T - I|| = {err:.1e})")

# 2. basis="frame" is exactly the original construction
original = construct_real_tff(n // 2, 1, n // 2)[0].view(-1, n)
assert torch.equal(build_basis("frame", n, 2, 0), original)
print("ok  frame basis is bit-identical to the original construct_real_tff")

# 3. building the random basis leaves the global RNG alone, so head init and data order match across arms
torch.manual_seed(0); expected = torch.rand(3)
torch.manual_seed(0); build_basis("random", n, 2, basis_seed=0); got = torch.rand(3)
assert torch.equal(expected, got)
print("ok  random basis does not consume the global RNG")

# 4. same coefficients -> delta W of the same size under every basis; frame delta W matches the original formula
norms = {}
for basis in ("frame", "random", "identity"):
    frame_layer.tffs.clear(); frame_layer._tffs_on_device.clear()
    torch.manual_seed(42)
    lin = Linear(nn.Linear(n, n), "default", n_ff_coeffs=1000, scale=10.0, tff_l=2, tff_block_size=768,
                 share_entry=True, basis=basis)
    dw = lin.get_delta_weight("default").double()
    norms[basis] = torch.linalg.norm(dw).item()
    if basis == "frame":
        idx, c = lin.indices["default"], lin.ff_coeffs["default"].detach().double()
        S = torch.zeros(n, n, dtype=torch.float64); S[idx[0], idx[1]] = c
        F = original.double()
        ref = (F.T @ S @ F * 10.0 / math.sqrt(n * n)).T
        assert torch.allclose(dw, ref, atol=1e-5), "frame delta W differs from the original formula"
        print("ok  frame delta W matches F^T S F * scale / n")
print("ok  ||delta W||_F per basis:", {k: round(v, 4) for k, v in norms.items()})
assert max(norms.values()) - min(norms.values()) < 1e-3 * max(norms.values())
print("all checks passed")
