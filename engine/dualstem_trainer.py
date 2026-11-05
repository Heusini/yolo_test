from pathlib import Path

import evrgb  # noqa: F401  (registers DualStemFuse for yaml parsing)
from evrgb.config import load_dualstem_cfg
from evrgb.temporal import TemporalBlock, clear_memory
from evrgb.weights import load_pretrained, stem_kind
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import RANK

from engine.eventrgbtrainer import EventRGBTrainer


class DualStemTrainer(EventRGBTrainer):
    """EventRGBTrainer for the dual-stem and two-tower models: resolves the yaml for the scale and remaps weights."""

    # Modality dropout (train only), overrides the yaml literals when not None.
    P_DROP_RGB = None
    P_DROP_EVT = None
    # "none" or "lstm": temporal block in the event stem (DualStemFuse.evt_mem), identity at init; train with CLIP_LEN > 1.
    MEM = "none"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(load_dualstem_cfg(cfg), nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if self.MEM == "lstm":
            model.model[0].evt_mem = TemporalBlock(model.model[0].mem_channels)
        else:
            assert self.MEM == "none", self.MEM
        if self.P_DROP_RGB is not None:
            model.model[0].p_drop_rgb = self.P_DROP_RGB
        if self.P_DROP_EVT is not None:
            model.model[0].p_drop_evt = self.P_DROP_EVT
        src = weights
        if isinstance(self.args.pretrained, (str, Path)):  # explicit checkpoint wins (e.g. pretrained="./yolo26n.pt")
            src, _ = load_checkpoint(str(self.args.pretrained))
        if src is not None:
            sd = src if isinstance(src, dict) else src.float().state_dict()
            if stem_kind(sd) == stem_kind(model.state_dict()):
                model.load(src)  # same layout (resume / phase 2): plain name+shape copy
            else:
                load_pretrained(model, src)
        clear_memory(model)
        return model
