from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def index_mask_to_color_mask(
    index_mask: np.ndarray,
    class_id_to_color: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    h, w = index_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in class_id_to_color.items():
        out[index_mask == cls_id] = color
    return out


def make_color_overlay(
    image_rgb: np.ndarray,
    color_mask_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    image_rgb = image_rgb.astype(np.float32)
    color_mask_rgb = color_mask_rgb.astype(np.float32)
    overlay = image_rgb * (1 - alpha) + color_mask_rgb * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_single_mask_overlay(
    image_rgb: np.ndarray,
    mask_255: np.ndarray,
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    result = image_rgb.copy()
    overlay = result.copy()
    overlay[mask_255 > 0] = color
    result = cv2.addWeighted(result, 1.0 - alpha, overlay, alpha, 0)

    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(
                result,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
    return result
