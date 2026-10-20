"""TemporalBlock: identity at init, state carried across calls, per-slot reset, shape change, gradient through time.

usage: PYTHONPATH=. python test/test_temporal.py
"""

import torch

from evrgb.temporal import TemporalBlock

torch.manual_seed(0)
C, H, W = 8, 6, 10


def randomize(block):
    for p in block.parameters():
        p.data.normal_(0, 0.3)
    return block


def main():
    blk = TemporalBlock(C)
    x = torch.randn(6, C, H, W)
    assert torch.equal(blk(x), x), "not identity at init (L=1)"
    blk.clip_len = 3
    assert torch.equal(blk(x), x), "not identity at init (L=3)"
    assert blk._h.shape == (2, C, H, W), blk._h.shape

    blk = randomize(TemporalBlock(C))
    blk.clip_len = 3
    fresh = randomize(TemporalBlock(C))
    fresh.load_state_dict(blk.state_dict())
    fresh.clip_len = 3
    x1, x2 = torch.randn(6, C, H, W), torch.randn(6, C, H, W)
    blk(x1)
    y2 = blk(x2)
    assert not torch.allclose(y2, fresh(x2)), "state not carried across calls"
    blk.reset()
    fresh.reset()
    assert torch.allclose(blk(x2), fresh(x2)), "reset() did not clear the state"

    blk(x1)
    fresh.reset()
    blk.reset(torch.tensor([True, False]))
    y = blk(x2)
    z = fresh(x2)
    assert torch.allclose(y[:3], z[:3]), "slot 0 not reset"
    assert not torch.allclose(y[3:], z[3:]), "slot 1 was reset too"

    blk.clip_len = 1
    blk(torch.randn(2, C, H, W))
    blk(torch.randn(5, C, H, W))  # batch-size change recreates the state
    blk.clip_len = 4
    blk(torch.randn(4, C, H, W).half().float())
    assert blk._h.shape == (1, C, H, W)
    blk.clear()
    assert blk._h is None and blk.clip_len == 1, "clear() must drop state and clip_len (both are pickled otherwise)"
    assert "_h" not in blk.state_dict() and all("_" not in k.split(".")[0] for k in blk.state_dict()), "state leaked into state_dict"

    blk = randomize(TemporalBlock(C))
    blk.clip_len = 4
    x = torch.randn(4, C, H, W, requires_grad=True)
    blk(x)[3].sum().backward()  # loss on the last frame only
    assert x.grad[0].abs().sum() > 0, "no gradient from frame 3 back to frame 0"
    print("ok")


if __name__ == "__main__":
    main()
