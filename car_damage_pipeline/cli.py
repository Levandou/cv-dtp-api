from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline: largest car -> quality gate -> parts segmentation")
    parser.add_argument("--image", required=True, help="Путь к входному изображению")
    parser.add_argument("--car-model", required=True, help="Путь к модели сегментации авто")
    parser.add_argument("--parts-model", required=True, help="Путь к модели сегментации деталей")
    parser.add_argument("--output-dir", default="output", help="Куда сохранить результаты")
    parser.add_argument("--image-size", type=int, default=512, help="Размер входа")
    parser.add_argument("--encoder-name", default="resnet34", help="Имя encoder")
    parser.add_argument("--encoder-weights", default="imagenet", help="Pretrained encoder weights")
    parser.add_argument("--num-classes", type=int, default=40, help="Число классов у модели деталей")
    parser.add_argument("--device", default=None, help="cuda или cpu")
    parser.add_argument("--car-conf", type=float, default=0.20, help="conf threshold")
    parser.add_argument("--car-iou", type=float, default=0.50, help="iou threshold")
    parser.add_argument(
        "--no-car-crop",
        action="store_true",
        help="Если указать, то запуск на всём кадре",
    )
    return parser
