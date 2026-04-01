from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .adaptive_photo_quality_gate import AdaptivePhotoQualityGate
from .class_maps import CLASS_ID_TO_COLOR, IDX_TO_CLASS
from .part_segmenter import PartSegmenter
from .geometry_utils import clip_box_to_image, scale_box
from .image_utils import draw_single_mask_overlay, index_mask_to_color_mask, make_color_overlay
from .largest_car_finder import LargestCarFinder
from .largest_car_result import LargestCarResult


class CarPartsPipeline:
    def __init__(
        self,
        car_segmenter: LargestCarFinder,
        quality_gate: AdaptivePhotoQualityGate,
        parts_segmenter: PartSegmenter,
        use_car_crop_for_parts: bool = True,
    ):
        self.car_segmenter = car_segmenter
        self.quality_gate = quality_gate
        self.parts_segmenter = parts_segmenter
        self.use_car_crop_for_parts = use_car_crop_for_parts

    def run(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {image_path}")

        original_h, original_w = original_bgr.shape[:2]

        largest_car = self.car_segmenter.find_largest_car(original_bgr)

        quality_before = self.quality_gate.evaluate(original_bgr)
        processed_bgr, applied_ops = self.quality_gate.preprocess(original_bgr, quality_before)
        quality_after = self.quality_gate.evaluate(processed_bgr)

        processed_h, processed_w = processed_bgr.shape[:2]
        scaled_car = self._resize_largest_car_result(
            largest_car,
            original_size=(original_w, original_h),
            new_size=(processed_w, processed_h),
        )

        processed_path = output_path / "01_processed_image.jpg"
        cv2.imwrite(str(processed_path), processed_bgr)

        report: Dict[str, Any] = {
            "image_path": str(Path(image_path).resolve()),
            "before_quality": quality_before,
            "applied_ops": applied_ops,
            "after_quality": quality_after,
            "largest_car": {
                "found": scaled_car.found,
                "bbox_xyxy": list(scaled_car.bbox_xyxy) if scaled_car.bbox_xyxy else None,
                "confidence": scaled_car.confidence,
                "area_pixels": scaled_car.area_pixels,
            },
            "parts_segmentation": {
                "executed": False,
                "reason": None,
                "detected_classes": [],
                "detected_class_names": [],
                "used_crop": self.use_car_crop_for_parts,
            },
            "files": {
                "processed_image": str(processed_path),
            },
        }

        if scaled_car.found and scaled_car.mask is not None:
            largest_mask_path = output_path / "02_largest_car_mask.png"
            largest_overlay_path = output_path / "03_largest_car_overlay.png"
            cv2.imwrite(str(largest_mask_path), scaled_car.mask)
            overlay = draw_single_mask_overlay(
                image_rgb=cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB),
                mask_255=scaled_car.mask,
                bbox_xyxy=scaled_car.bbox_xyxy,
                label="largest car",
                color=(255, 0, 0),
                alpha=0.35,
            )
            cv2.imwrite(str(largest_overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            report["files"]["largest_car_mask"] = str(largest_mask_path)
            report["files"]["largest_car_overlay"] = str(largest_overlay_path)

        if not scaled_car.found or scaled_car.mask is None or scaled_car.bbox_xyxy is None:
            report["parts_segmentation"]["reason"] = "largest_car_not_found"
            return self._save_report(output_path, report)

        if quality_after["status"] != "PASS":
            report["parts_segmentation"]["reason"] = f"quality_after_preprocessing_is_{quality_after['status']}"
            return self._save_report(output_path, report)

        pred_mask_full, pred_color_full = self._run_parts_segmentation(processed_bgr, scaled_car)
        pred_mask_full[scaled_car.mask == 0] = 0
        pred_color_full = index_mask_to_color_mask(pred_mask_full, CLASS_ID_TO_COLOR)
        overlay = make_color_overlay(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB), pred_color_full, alpha=0.45)

        mask_path = output_path / "04_parts_mask_class_ids.png"
        color_mask_path = output_path / "05_parts_color_mask.png"
        overlay_path = output_path / "06_parts_overlay.png"
        cv2.imwrite(str(mask_path), pred_mask_full)
        cv2.imwrite(str(color_mask_path), cv2.cvtColor(pred_color_full, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        unique_classes = [int(x) for x in np.unique(pred_mask_full) if int(x) != 0]
        report["parts_segmentation"]["executed"] = True
        report["parts_segmentation"]["reason"] = "ok"
        report["parts_segmentation"]["detected_classes"] = unique_classes
        report["parts_segmentation"]["detected_class_names"] = [IDX_TO_CLASS.get(x, f"unknown_{x}") for x in unique_classes]
        report["files"]["parts_mask_class_ids"] = str(mask_path)
        report["files"]["parts_color_mask"] = str(color_mask_path)
        report["files"]["parts_overlay"] = str(overlay_path)

        return self._save_report(output_path, report)

    def _run_parts_segmentation(
        self,
        processed_bgr: np.ndarray,
        largest_car: LargestCarResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = processed_bgr.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_color = np.zeros((h, w, 3), dtype=np.uint8)

        if not self.use_car_crop_for_parts:
            image_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
            pred_mask, pred_color = self.parts_segmenter.predict_rgb(image_rgb)
            return pred_mask, pred_color

        x1, y1, x2, y2 = largest_car.bbox_xyxy
        x1, y1, x2, y2 = clip_box_to_image((x1, y1, x2, y2), w, h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Некорректный bbox самой большой машины после масштабирования.")

        crop_bgr = processed_bgr[y1:y2, x1:x2]
        crop_mask = largest_car.mask[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        pred_crop_mask, _ = self.parts_segmenter.predict_rgb(crop_rgb)
        pred_crop_mask[crop_mask == 0] = 0
        pred_crop_color = index_mask_to_color_mask(pred_crop_mask, CLASS_ID_TO_COLOR)

        full_mask[y1:y2, x1:x2] = pred_crop_mask
        full_color[y1:y2, x1:x2] = pred_crop_color
        return full_mask, full_color

    def _resize_largest_car_result(
        self,
        largest_car: LargestCarResult,
        original_size: Tuple[int, int],
        new_size: Tuple[int, int],
    ) -> LargestCarResult:
        if not largest_car.found or largest_car.mask is None or largest_car.bbox_xyxy is None:
            return largest_car

        orig_w, orig_h = original_size
        new_w, new_h = new_size

        if (orig_w, orig_h) == (new_w, new_h):
            return largest_car

        resized_mask = cv2.resize(largest_car.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        x1, y1, x2, y2 = scale_box(largest_car.bbox_xyxy, orig_w, orig_h, new_w, new_h)
        x1, y1, x2, y2 = clip_box_to_image((x1, y1, x2, y2), new_w, new_h)

        return LargestCarResult(
            found=True,
            mask=resized_mask.astype(np.uint8),
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=largest_car.confidence,
            area_pixels=int((resized_mask > 0).sum()),
        )

    @staticmethod
    def _save_report(output_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
        report_path = output_path / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["files"]["report_json"] = str(report_path)
        return report
