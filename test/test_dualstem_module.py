"""CPU check of DualStemFuse. usage (repo root): PYTHONPATH=. python test/test_dualstem_module.py"""

import torch

from evrgb import DualStemFuse


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 13, 384, 640)

    m = DualStemFuse(13, 3, 64).eval()
    fused, f_rgb, f_evt, g = m(x)
    assert fused.shape == f_rgb.shape == f_evt.shape == g.shape == (2, 64, 48, 80), fused.shape
    assert abs(g.mean().item() - 0.5) < 1e-6 and g.min() == g.max(), "gate must start at exactly 0.5"
    assert torch.allclose(fused, 0.5 * f_rgb + 0.5 * f_evt), "fused must be the plain average at init"

    # eval never drops; train drops the RGB input with p_drop_rgb
    m = DualStemFuse(13, 3, 64, p_drop_rgb=1.0)
    m.eval()
    assert torch.equal(m(x)[1], m.rgb_stem(x[:, :3])), "eval mode must not drop"
    m.train()
    assert torch.equal(m(x)[1], m.rgb_stem(torch.zeros_like(x[:, :3]))), "train mode must zero the RGB input"

    n_params = sum(p.numel() for p in DualStemFuse(13, 3, 64).parameters())
    print(f"ok: shapes {tuple(fused.shape)}, gate init 0.5, dropout ok, params {n_params/1e3:.1f}k")


if __name__ == "__main__":
    main()
