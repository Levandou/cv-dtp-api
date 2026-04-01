from __future__ import annotations

from typing import Tuple


def scale_box(
    box_xyxy: Tuple[int, int, int, int],
    old_w: int,
    old_h: int,
    new_w: int,
    new_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def clip_box_to_image(
    box_xyxy: Tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w))
    y2 = max(0, min(y2, image_h))
    return x1, y1, x2, y2
