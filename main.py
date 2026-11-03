import argparse
from pathlib import Path
from traffic_yolo.config import load_config
from traffic_yolo.pipeline import TrafficMonitorPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrafficYOLO - Automated Traffic Monitoring and Violation Detection System"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs and evidence (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Computation device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable annotated video output recording to boost processing speed.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.device:
        config["models"]["device"] = args.device
    if args.conf is not None:
        config["inference"]["confidence"] = args.conf

    pipeline = TrafficMonitorPipeline(config)

    save_video = not args.no_video if args.no_video else None
    summary = pipeline.run(
        video_source=args.video,
        output_dir=args.output_dir,
        save_video=save_video,
    )

    print("\n" + "=" * 45)
    print("           TRAFFIC ANALYSIS SUMMARY          ")
    print("=" * 45)
    print(f"Total Frames Processed: {summary['total_frames']}")
    print("Vehicle Counts:")
    for v_type, cnt in summary["counts"].items():
        print(f"  - {v_type.capitalize()}: {cnt}")
    print(f"Total Red-Light Violations: {summary['violations_count']}")
    if summary["output_video"]:
        print(f"Output Video: {summary['output_video']}")
    if summary["output_csv"]:
        print(f"CSV Report:   {summary['output_csv']}")
    print(f"Evidence Dir: {summary['evidence_dir']}")
    print("=" * 45)


if __name__ == "__main__":
    main()
