from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from .geometry_utils import clip_box_to_image
from .largest_car_result import LargestCarResult


class LargestCarFinder:
    def __init__(
        self,
        model_path: str,
        car_class_id: int = 2,
        conf: float = 0.20,
        iou: float = 0.50,
    ):
        self.model = YOLO(model_path)
        self.car_class_id = car_class_id
        self.conf = conf
        self.iou = iou

    def find_largest_car(self, image_bgr: np.ndarray) -> LargestCarResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=image_rgb,
            classes=[self.car_class_id],
            conf=self.conf,
            iou=self.iou,
            retina_masks=True,
            verbose=False,
        )

        if not results:
            return LargestCarResult(False, None, None, None, None)

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return LargestCarResult(False, None, None, None, None)

        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        scores = r.boxes.conf.detach().cpu().numpy().astype(float)
        masks = r.masks.data.detach().cpu().numpy()

        img_h, img_w = image_bgr.shape[:2]
        if masks.shape[1] != img_h or masks.shape[2] != img_w:
            resized_masks = []
            for m in masks:
                resized_masks.append(cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST))
            masks = np.stack(resized_masks, axis=0)

        masks_bin = (masks > 0.5).astype(np.uint8)
        areas = masks_bin.reshape(masks_bin.shape[0], -1).sum(axis=1)
        largest_idx = int(np.argmax(areas))

        mask = masks_bin[largest_idx] * 255
        bbox = tuple(int(v) for v in boxes[largest_idx].tolist())
        x1, y1, x2, y2 = clip_box_to_image(bbox, img_w, img_h)
        return LargestCarResult(
            found=True,
            mask=mask.astype(np.uint8),
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=float(scores[largest_idx]),
            area_pixels=int(areas[largest_idx]),
        )
