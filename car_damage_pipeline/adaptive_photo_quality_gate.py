from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class AdaptivePhotoQualityGate:
    def __init__(
        self,
        min_width: int = 900,
        min_height: int = 700,
        target_short_side: int = 1200,
        min_contrast_std: float = 28.0,
        blur_fix_threshold: float = 90.0,
        blur_reject_threshold: float = 60.0,
        dark_ratio_fix_threshold: float = 0.35,
        dark_ratio_reject_threshold: float = 0.75,
        bright_ratio_fix_threshold: float = 0.08,
        bright_ratio_reject_threshold: float = 0.22,
        glare_ratio_fix_threshold: float = 0.010,
        glare_ratio_reject_threshold: float = 0.040,
        noise_fix_threshold: float = 10.0,
        noise_reject_threshold: float = 20.0,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.target_short_side = target_short_side
        self.min_contrast_std = min_contrast_std
        self.blur_fix_threshold = blur_fix_threshold
        self.blur_reject_threshold = blur_reject_threshold
        self.dark_ratio_fix_threshold = dark_ratio_fix_threshold
        self.dark_ratio_reject_threshold = dark_ratio_reject_threshold
        self.bright_ratio_fix_threshold = bright_ratio_fix_threshold
        self.bright_ratio_reject_threshold = bright_ratio_reject_threshold
        self.glare_ratio_fix_threshold = glare_ratio_fix_threshold
        self.glare_ratio_reject_threshold = glare_ratio_reject_threshold
        self.noise_fix_threshold = noise_fix_threshold
        self.noise_reject_threshold = noise_reject_threshold

    @staticmethod
    def _round(x: float) -> float:
        return float(np.round(x, 6))

    @staticmethod
    def _gamma_correct(image_bgr: np.ndarray, gamma: float = 1.4) -> np.ndarray:
        gamma = max(gamma, 1e-6)
        lut = np.array(
            [np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image_bgr, lut)

    @staticmethod
    def _apply_clahe_lab(
        image_bgr: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _denoise(image_bgr: np.ndarray, h: int = 6, h_color: int = 6) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image_bgr, None, h, h_color, 7, 21)

    @staticmethod
    def _mild_unsharp_mask(image_bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
        blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _compress_highlights(image_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        v_norm = v / 255.0
        threshold = 0.78
        v_new = np.where(v_norm > threshold, threshold + (v_norm - threshold) * strength, v_norm)
        v_new = np.clip(v_new * 255.0, 0, 255).astype(np.uint8)
        hsv_new = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v_new])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    def _upscale_if_needed(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        h, w = image_bgr.shape[:2]
        short_side = min(h, w)
        if short_side >= self.target_short_side:
            return image_bgr, None

        scale = min(self.target_short_side / float(short_side), 2.0)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        upscaled = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled, {"op": "upscale", "scale": float(np.round(scale, 3)), "new_size": [new_w, new_h]}

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        median = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, median)
        edges = cv2.Canny(gray, 50, 150)
        flat_mask = edges == 0
        values = residual[flat_mask] if np.count_nonzero(flat_mask) > 500 else residual.reshape(-1)
        return float(values.std())

    @staticmethod
    def _compute_glare_ratio(image_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        glare_mask = ((v > 245) & (s < 40)).astype(np.uint8) * 255
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(glare_mask, 8)

        area_total = image_bgr.shape[0] * image_bgr.shape[1]
        glare_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= max(30, int(0.0003 * area_total)):
                glare_area += area
        return glare_area / float(area_total)

    def evaluate(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if image_bgr is None:
            raise ValueError("Изображение не загружено.")

        h, w = image_bgr.shape[:2]
        area = h * w
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark_ratio = float(np.mean(gray < 40))
        bright_ratio = float(np.mean(gray > 245))
        glare_ratio = float(self._compute_glare_ratio(image_bgr))
        noise_std = float(self._estimate_noise(gray))
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 100, 200))) / float(area)
        short_side = min(h, w)
        size_ok = (w >= self.min_width) and (h >= self.min_height)

        empty_like = contrast_std < 8.0 and edge_density < 0.003
        severe_blur = laplacian_variance < self.blur_reject_threshold
        mild_blur = self.blur_reject_threshold <= laplacian_variance < self.blur_fix_threshold
        severe_dark = dark_ratio > self.dark_ratio_reject_threshold
        mild_dark = self.dark_ratio_fix_threshold < dark_ratio <= self.dark_ratio_reject_threshold
        severe_bright = bright_ratio > self.bright_ratio_reject_threshold
        mild_bright = self.bright_ratio_fix_threshold < bright_ratio <= self.bright_ratio_reject_threshold
        severe_glare = glare_ratio > self.glare_ratio_reject_threshold
        mild_glare = self.glare_ratio_fix_threshold < glare_ratio <= self.glare_ratio_reject_threshold
        severe_noise = noise_std > self.noise_reject_threshold
        mild_noise = self.noise_fix_threshold < noise_std <= self.noise_reject_threshold
        low_contrast = contrast_std < self.min_contrast_std
        too_small = short_side < self.target_short_side

        checks = {
            "readable": {"passed": not empty_like, "message": f"contrast_std={contrast_std:.2f}, edge_density={edge_density:.4f}"},
            "blur": {"passed": not severe_blur, "message": f"laplacian_variance={laplacian_variance:.2f}"},
            "overexposure": {"passed": not severe_bright, "message": f"bright_ratio={bright_ratio:.4f}"},
            "glare": {"passed": not severe_glare, "message": f"glare_ratio={glare_ratio:.4f}"},
            "noise": {"passed": not severe_noise, "message": f"noise_std={noise_std:.2f}"},
            "size": {"passed": size_ok, "message": f"width={w}, height={h}, short_side={short_side}"},
            "contrast": {"passed": not low_contrast, "message": f"contrast_std={contrast_std:.2f}"},
            "darkness": {"passed": not severe_dark, "message": f"dark_ratio={dark_ratio:.4f}"},
        }

        reject_reasons: List[str] = []
        fix_reasons: List[str] = []

        if empty_like:
            reject_reasons.append("photo_not_readable_or_almost_empty")
        if severe_blur:
            reject_reasons.append("severe_blur")
        if severe_bright:
            reject_reasons.append("severe_overexposure")
        if severe_glare:
            reject_reasons.append("strong_glare")
        if severe_dark:
            reject_reasons.append("severe_underexposure")

        if mild_blur:
            fix_reasons.append("mild_blur")
        if mild_dark:
            fix_reasons.append("mild_underexposure")
        if mild_bright:
            fix_reasons.append("moderate_overexposure")
        if mild_glare:
            fix_reasons.append("moderate_glare")
        if mild_noise:
            fix_reasons.append("visible_noise")
        if low_contrast:
            fix_reasons.append("low_contrast")
        if too_small:
            fix_reasons.append("small_scale")

        if reject_reasons:
            status = "REJECT"
        elif fix_reasons:
            status = "FIX_AND_RETRY"
        else:
            status = "PASS"

        metrics = {
            "width": int(w),
            "height": int(h),
            "short_side": int(short_side),
            "brightness_mean": self._round(brightness_mean),
            "contrast_std": self._round(contrast_std),
            "laplacian_variance": self._round(laplacian_variance),
            "dark_ratio": self._round(dark_ratio),
            "bright_ratio": self._round(bright_ratio),
            "glare_ratio": self._round(glare_ratio),
            "noise_std": self._round(noise_std),
            "edge_density": self._round(edge_density),
        }

        return {
            "status": status,
            "metrics": metrics,
            "checks": checks,
            "reject_reasons": reject_reasons,
            "fix_reasons": fix_reasons,
        }

    def preprocess(self, image_bgr: np.ndarray, evaluation: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        processed = image_bgr.copy()
        applied_ops: List[Dict[str, Any]] = []
        status = evaluation["status"]
        metrics = evaluation["metrics"]
        fix_reasons = set(evaluation["fix_reasons"])

        if status == "REJECT":
            return processed, applied_ops

        if "small_scale" in fix_reasons:
            processed, op = self._upscale_if_needed(processed)
            if op is not None:
                applied_ops.append(op)
        if "visible_noise" in fix_reasons:
            processed = self._denoise(processed, h=6, h_color=6)
            applied_ops.append({"op": "denoise_nlm", "h": 6, "h_color": 6})
        if "mild_underexposure" in fix_reasons:
            dark_ratio = metrics["dark_ratio"]
            gamma = 1.35 if dark_ratio < 0.50 else 1.60
            processed = self._gamma_correct(processed, gamma=gamma)
            applied_ops.append({"op": "gamma_brighten", "gamma": gamma})
        if "moderate_overexposure" in fix_reasons or "moderate_glare" in fix_reasons:
            processed = self._compress_highlights(processed, strength=0.55)
            applied_ops.append({"op": "highlight_compression", "strength": 0.55})
        if "low_contrast" in fix_reasons:
            processed = self._apply_clahe_lab(processed, clip_limit=2.0, tile_grid_size=(8, 8))
            applied_ops.append({"op": "clahe_lab", "clip_limit": 2.0, "tile_grid_size": [8, 8]})
        if "mild_blur" in fix_reasons:
            processed = self._mild_unsharp_mask(processed, sigma=1.0, amount=0.5)
            applied_ops.append({"op": "mild_unsharp_mask", "sigma": 1.0, "amount": 0.5})

        return processed, applied_ops
