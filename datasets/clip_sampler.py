"""Batch sampler for recurrent training: a batch = `slots` sequence streams x `clip_len` consecutive frames.

Stream slots (RVT's TBPTT loader): every sequence is cut into clips of clip_len frames starting at pos 0 (tail shorter
than clip_len dropped); sequences are dealt to the stream slot with the fewest clips; batch k is clip k of every slot,
so slot b in batch k+1 continues the sequence slot b had in batch k, and a slot starts a new sequence exactly when its
first frame has pos 0. The first `random_slots` slots (RVT's BPTT loader) instead draw an independent random clip every
batch and must always be reset. Indices are slot-major; random slots come first.
__iter__ re-plans with an internal epoch counter (InfiniteDataLoader re-calls it, possibly early via prefetch).
"""

import random

from torch.utils.data import Sampler


class ClipBatchSampler(Sampler):
    def __init__(self, match_list, slots: int, clip_len: int, shuffle: bool = False, seed: int = 0, random_slots: int = 0):
        assert slots >= 1 and clip_len >= 1 and 0 <= random_slots <= slots
        self.slots, self.clip_len, self.shuffle, self.seed = slots, clip_len, shuffle, seed
        self.random_slots, self.stream_slots = random_slots, slots - random_slots
        self.epoch = 0
        clips = {}
        self.starts = []  # every index whose clip fits inside its sequence (random slots)
        for i, m in enumerate(match_list):
            if m.pos + clip_len <= m.seq_len:
                self.starts.append(i)
                if m.pos % clip_len == 0:
                    clips.setdefault(m.seq, []).append(i)
        self.seq_clips = [clips[s] for s in sorted(clips)]  # per sequence: global start index of each clip
        assert self.seq_clips, f"no sequence has {clip_len} frames"
        self.stream_slots = min(self.stream_slots, len(self.seq_clips))  # fewer sequences than slots: smaller batch
        self.slots = self.random_slots + self.stream_slots
        self._len = sum(len(c) for c in self.seq_clips) // max(self.random_slots, 1)  # random-only case
        if self.stream_slots:
            self._len = self._plan(0)[1]

    def _plan(self, epoch):
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.seq_clips)))
        if self.shuffle:
            rng.shuffle(order)
        streams = [[] for _ in range(self.stream_slots)]
        for s in order:
            if streams:
                streams[min(range(self.stream_slots), key=lambda k: len(streams[k]))].extend(self.seq_clips[s])
        n = min(len(st) for st in streams) if streams else self._len
        return [st[:n] for st in streams], n, rng

    def __iter__(self):
        streams, n, rng = self._plan(self.epoch)
        self.epoch += 1
        L = self.clip_len
        for k in range(n):
            heads = [rng.choice(self.starts) for _ in range(self.random_slots)] + [st[k] for st in streams]
            yield [i for h in heads for i in range(h, h + L)]

    def __len__(self):
        return self._len
