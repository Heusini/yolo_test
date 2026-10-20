"""Two-stem event+RGB front end for Ultralytics YOLO. `import evrgb` registers the modules for YAML parsing."""

from evrgb.modules import DualStemFuse, GatedFuse, register
from evrgb.temporal import TemporalBlock

register()

__all__ = ["DualStemFuse", "GatedFuse", "TemporalBlock", "register"]
