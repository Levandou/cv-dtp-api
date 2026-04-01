from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LargestCarResult:
    found: bool
    mask: Optional[np.ndarray]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    confidence: Optional[float]
    area_pixels: Optional[int]
