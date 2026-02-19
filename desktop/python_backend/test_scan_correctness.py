"""
Scan correctness test — verifies chunked selective_scan matches
the original sequential scan exactly.

Run: python test_scan_correctness.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ── Original sequential scan (the known-correct reference) ──
def selective_scan_reference(x, delta, A, B, C, D_skip):
    """Naive sequential scan — the original implementation."""
    B_batch, L, D_dim = x.shape
    N = A.shape[1]
    delta_A = torch.exp(delta.unsqueeze(-1) * A)
    delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)
    y = torch.empty(B_batch, L, D_dim, device=x.device, dtype=x.dtype)
    h = torch.zeros(B_batch, D_dim, N, device=x.device, dtype=x.dtype)
    for t in range(L):
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
        y[:, t] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
    y = y + x * D_skip
    return y

# ── Import the chunked version from mamba_model ──
from mamba_model import selective_scan as selective_scan_chunked

def test_correctness():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Match real training dimensions: SmallJamba
    B, L, D, N = 4, 120, 512, 16  # d_inner = 256*2 = 512

    # Create deterministic inputs
    x = torch.randn(B, L, D, device=device)
    delta = torch.randn(B, L, D, device=device).abs() * 0.1  # positive deltas
    A = -torch.rand(D, N, device=device) * 2  # negative A (stable)
    B_param = torch.randn(B, L, N, device=device) * 0.1
    C_param = torch.randn(B, L, N, device=device) * 0.1
    D_skip = torch.ones(D, device=device)

    # ── Test 1: fp32 baseline ──
    with torch.no_grad():
        y_ref = selective_scan_reference(x, delta, A, B_param, C_param, D_skip)
        y_chunked = selective_scan_chunked(x, delta, A, B_param, C_param, D_skip)

    max_abs = (y_ref - y_chunked).abs().max().item()
    mean_abs = (y_ref - y_chunked).abs().mean().item()
    rel_err = ((y_ref - y_chunked).abs() / (y_ref.abs() + 1e-8)).max().item()

    print(f"\n{'='*50}")
    print(f"FP32 Correctness Test")
    print(f"  Max absolute error:  {max_abs:.2e}")
    print(f"  Mean absolute error: {mean_abs:.2e}")
    print(f"  Max relative error:  {rel_err:.2e}")

    fp32_pass = max_abs < 1e-5
    print(f"  PASS: {fp32_pass} (threshold: 1e-5)")

    # ── Test 2: fp16 autocast (matches training) ──
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        x16 = x.half()
        delta16 = delta.half()
        A16 = A.half()
        B16 = B_param.half()
        C16 = C_param.half()
        D16 = D_skip.half()

        y_ref16 = selective_scan_reference(x16, delta16, A16, B16, C16, D16)
        y_chunked16 = selective_scan_chunked(x16, delta16, A16, B16, C16, D16)

    max_abs16 = (y_ref16.float() - y_chunked16.float()).abs().max().item()
    rel_err16 = ((y_ref16.float() - y_chunked16.float()).abs() / (y_ref16.float().abs() + 1e-6)).max().item()

    print(f"\n{'='*50}")
    print(f"FP16 Correctness Test (matches AMP training)")
    print(f"  Max absolute error:  {max_abs16:.2e}")
    print(f"  Max relative error:  {rel_err16:.2e}")

    fp16_pass = max_abs16 < 1e-3
    print(f"  PASS: {fp16_pass} (threshold: 1e-3)")

    # ── Test 3: Check error doesn't grow across timesteps ──
    per_step_err = (y_ref - y_chunked).abs().mean(dim=(0, 2))  # (L,)
    early_err = per_step_err[:20].mean().item()
    late_err = per_step_err[-20:].mean().item()

    print(f"\n{'='*50}")
    print(f"Error propagation check")
    print(f"  Mean error (first 20 steps): {early_err:.2e}")
    print(f"  Mean error (last 20 steps):  {late_err:.2e}")
    print(f"  Ratio (late/early):          {late_err/(early_err+1e-12):.2f}x")

    no_explosion = late_err < early_err * 10
    print(f"  PASS: {no_explosion} (late < 10x early)")

    # ── Summary ──
    all_pass = fp32_pass and fp16_pass and no_explosion
    print(f"\n{'='*50}")
    print(f"{'✅ ALL TESTS PASSED' if all_pass else '❌ TESTS FAILED'}")
    print(f"{'='*50}")
    return all_pass

if __name__ == "__main__":
    test_correctness()
