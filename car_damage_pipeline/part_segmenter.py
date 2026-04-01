from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

from .class_maps import CLASS_ID_TO_COLOR
from .image_utils import index_mask_to_color_mask


class PartSegmenter:
    def __init__(
        self,
        checkpoint_path: str,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        image_size: int = 512,
        device: Optional[str] = None,
        num_classes: int = 40,
    ):
        self.checkpoint_path = checkpoint_path
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.image_size = image_size
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.preprocess_input = get_preprocessing_fn(self.encoder_name, pretrained=self.encoder_weights)

    def _build_architecture(self) -> torch.nn.Module:
        return smp.DeepLabV3Plus(
            encoder_name=self.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=self.num_classes,
        )

    def _load_model(self) -> torch.nn.Module:
        obj = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, dict) and "state_dict" in obj:
            model = self._build_architecture()
            model.load_state_dict(obj["state_dict"], strict=True)
        elif isinstance(obj, dict):
            model = self._build_architecture()
            model.load_state_dict(obj, strict=True)
        else:
            raise TypeError(
                "Неизвестный формат checkpoint. Ожидалась полная модель torch.nn.Module, state_dict или словарь с ключом 'state_dict'."
            )

        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_rgb(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        resized = pil_img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        resized_np = np.array(resized)

        x = self.preprocess_input(resized_np).astype("float32")
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(x)
        pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_mask = np.array(
            Image.fromarray(pred_small).resize((w, h), resample=Image.NEAREST)
        ).astype(np.uint8)
        pred_color = index_mask_to_color_mask(pred_mask, CLASS_ID_TO_COLOR)
        return pred_mask, pred_color
