from pathlib import Path

import evrgb  # noqa: F401  (registers DualStemFuse for yaml parsing)
from evrgb.config import load_dualstem_cfg
from evrgb.weights import is_single_stem, load_pretrained
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import RANK

from engine.eventrgbtrainer import EventRGBTrainer


class DualStemTrainer(EventRGBTrainer):
    """EventRGBTrainer with the two-stem model: resolves the yaml for the scale and remaps single-stem weights."""

    # Modality dropout (train only), overrides the yaml literals when not None.
    P_DROP_RGB = None
    P_DROP_EVT = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(load_dualstem_cfg(cfg), nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if self.P_DROP_RGB is not None:
            model.model[0].p_drop_rgb = self.P_DROP_RGB
        if self.P_DROP_EVT is not None:
            model.model[0].p_drop_evt = self.P_DROP_EVT
        src = weights
        if isinstance(self.args.pretrained, (str, Path)):  # explicit checkpoint wins (e.g. pretrained="./yolo26n.pt")
            src, _ = load_checkpoint(str(self.args.pretrained))
        if src is not None:
            sd = src if isinstance(src, dict) else src.float().state_dict()
            if is_single_stem(sd):
                load_pretrained(model, src)
            else:
                model.load(src)  # dual-stem checkpoint (resume / phase 2): plain name+shape copy
        return model
