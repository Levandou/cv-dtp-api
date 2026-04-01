from __future__ import annotations

import json

from car_damage_pipeline.cli import build_arg_parser
from car_damage_pipeline.adaptive_photo_quality_gate import AdaptivePhotoQualityGate
from car_damage_pipeline.car_parts_pipeline import CarPartsPipeline
from car_damage_pipeline.part_segmenter import PartSegmenter
from car_damage_pipeline.largest_car_finder import LargestCarFinder


def main() -> None:
    args = build_arg_parser().parse_args()

    car_segmenter = LargestCarFinder(
        model_path=args.car_model,
        conf=args.car_conf,
        iou=args.car_iou,
    )
    quality_gate = AdaptivePhotoQualityGate()
    parts_segmenter = PartSegmenter(
        checkpoint_path=args.parts_model,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        image_size=args.image_size,
        device=args.device,
        num_classes=args.num_classes,
    )
    pipeline = CarPartsPipeline(
        car_segmenter=car_segmenter,
        quality_gate=quality_gate,
        parts_segmenter=parts_segmenter,
        use_car_crop_for_parts=not args.no_car_crop,
    )

    report = pipeline.run(image_path=args.image, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
