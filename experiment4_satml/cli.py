import argparse
from pathlib import Path

from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run KLRD-RS Experiment 4 SaTML statistical bigram suppression")
    parser.add_argument("--stage", choices=("all", "prepare", "calibration", "mapping",
        "generation", "verification", "arc", "report"), default="all")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/experiment4.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume from atomic stage journals and cached mapping shards")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved inputs without loading models or downloading assets")
    args = parser.parse_args(argv)
    if args.dry_run:
        import yaml
        path = Path(args.config).resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        print(f"config={path}")
        print(f"e3_full={config['paths']['e3_full_output']}")
        print(f"calibration_model={config['models']['calibration_model']}")
        print(f"mapping_batch_size={config['experiment']['calibration_batch_size']}")
        print(f"deltas={config['experiment']['deltas']}")
        return 0
    run(args.stage, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
