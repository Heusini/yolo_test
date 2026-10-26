"""Event-side rows of robustness_eval: bin shift semantics and count perturbations (CPU, no data needed)."""

import sys

import torch

sys.path[:0] = ["./", "./test"]
from robustness_eval import EVT_CUTOFF, EVT_DROP_P, EVT_NOISE_P, N_BINS, ShiftEvt, perturb

torch.manual_seed(0)
C, H, W = 2 * N_BINS, 8, 8
cur = torch.randint(0, 11, (C, H, W)).float()
nxt = torch.randint(0, 11, (C, H, W)).float()

# shift: k = 0 identity, k = N_BINS -> next frame, 0 < k < N_BINS -> newest k bins of each polarity come from nxt
assert torch.equal(ShiftEvt.shift(cur, nxt, 0), cur)
assert torch.equal(ShiftEvt.shift(cur, nxt, N_BINS), nxt)
k = 2
out = ShiftEvt.shift(cur, nxt, k)
for pol in range(2):
    o = out[pol * N_BINS : (pol + 1) * N_BINS]
    assert torch.equal(o[: N_BINS - k], cur[pol * N_BINS + k : (pol + 1) * N_BINS])
    assert torch.equal(o[N_BINS - k :], nxt[pol * N_BINS : pol * N_BINS + k])

# count perturbations act on the event channels only and stay within the cutoff
img = torch.cat([torch.rand(4, 3, H, W), torch.randint(0, 11, (4, C, H, W)).float()], 1)
big = torch.cat([torch.rand(4, 3, 64, 64), torch.randint(0, 11, (4, C, 64, 64)).float()], 1)
for cond in ["evt_half", "evt_double", "evt_noise", "evt_drop"]:
    out = perturb(img, cond)
    assert torch.equal(out[:, :3], img[:, :3]), cond
    assert out[:, 3:].min() >= 0 and out[:, 3:].max() <= EVT_CUTOFF, cond
assert torch.equal(perturb(img, "evt_half")[:, 3:], (img[:, 3:] * 0.5).round())
assert perturb(img, "evt_double")[:, 3:].max() == EVT_CUTOFF
added = (perturb(big, "evt_noise")[:, 3:] - big[:, 3:]).clamp(min=0)
assert abs(added.mean().item() - EVT_NOISE_P * (1 - 1 / 11)) < 0.002, added.mean()  # +1 where count < cutoff
kept = perturb(big, "evt_drop")[:, 3:]
assert (kept <= big[:, 3:]).all()
assert abs(kept.sum().item() / big[:, 3:].sum().item() - (1 - EVT_DROP_P)) < 0.01
print("ok")
