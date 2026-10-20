"""ClipBatchSampler on the fake set: slot-major contiguous clips, streams continue across batches, sequences start at pos 0.

usage: PYTHONPATH=. python test/test_clip_sampler.py [<dataset root>]
"""

import sys

from datasets.basedataset import Match, create_matching_items
from datasets.clip_sampler import ClipBatchSampler
from pathlib import Path


def synthetic(seq_lens):
    ml = []
    for s, n in enumerate(seq_lens):
        ml += [Match(f"e{s}_{i}", f"l{s}_{i}", f"r{s}_{i}", seq=s, pos=i, seq_len=n) for i in range(n)]
    return ml


def check(sampler, ml, slots, L, random_slots=0):
    prev = [None] * slots
    n_batches, starts = 0, 0
    for batch in sampler:
        assert len(batch) == slots * L
        for b in range(slots):
            frames = [ml[i] for i in batch[b * L:(b + 1) * L]]
            assert len({f.seq for f in frames}) == 1, "clip spans sequences"
            assert [f.pos for f in frames] == list(range(frames[0].pos, frames[0].pos + L)), "clip not contiguous"
            if b < random_slots:
                continue
            if frames[0].pos == 0:
                starts += 1
            else:
                assert prev[b] is not None and prev[b].seq == frames[0].seq and prev[b].pos + 1 == frames[0].pos, \
                    "stream broken across batches"
            prev[b] = frames[-1]
        n_batches += 1
    return n_batches, starts


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".cache/yolo_test_cpu/fake") / "train"
    ml = create_matching_items(root)
    seq_len = {m.seq: m.seq_len for m in ml}
    print(f"{len(ml)} frames, sequences: {seq_len}")
    for slots, L in [(2, 3), (1, 2), (2, 1), (1, 4)]:
        s = ClipBatchSampler(ml, slots, L, shuffle=True)
        nb, starts = check(s, ml, slots, L)
        assert nb == len(s), (nb, len(s))
        print(f"slots={slots} L={L}: {nb} batches, {starts} sequence starts, len={len(s)}")
    s = ClipBatchSampler(ml, 2, 3, shuffle=False)
    assert list(s) == list(s), "unshuffled order must be deterministic"

    ml = synthetic([17, 5, 9, 2, 12, 8, 30])  # tails of 1, 1, 1, (2 dropped entirely), 0, 0, 2 at L=4
    slots, L = 3, 4
    s = ClipBatchSampler(ml, slots, L, shuffle=True)
    nb, starts = check(s, ml, slots, L)
    assert 1 <= starts <= 6, starts  # 6 sequences have a clip; stream truncation may cut trailing ones
    print(f"synthetic {[m.seq_len for m in ml if m.pos == 0]}: slots={slots} L={L}: {nb} batches, {starts} starts, len={len(s)}")
    epochs = [list(s) for _ in range(4)]
    assert any(e != epochs[0] for e in epochs[1:]), "shuffle gives the same order every epoch"
    for e in epochs:
        check(iter(e), ml, slots, L)

    s = ClipBatchSampler(ml, 4, L, shuffle=True, random_slots=2)  # RVT "mixed": 2 random + 2 stream slots
    nb, starts = check(s, ml, 4, L, random_slots=2)
    heads = {tuple(ml[b[0]].pos for b in [batch[:L], batch[L:2 * L]]) for batch in s}
    assert any(p % L for h in heads for p in h), "random slots never start off the clip grid"
    print(f"mixed 2 random + 2 stream: {nb} batches, {starts} stream starts, len={len(s)}")
    s = ClipBatchSampler(ml, 2, L, random_slots=2)
    nb, _ = check(s, ml, 2, L, random_slots=2)
    assert nb == len(s) == 19 // 2, (nb, len(s))
    print("ok")


if __name__ == "__main__":
    main()
