"""Import-safe CLI for portable Llama Experiment 1 stages."""

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="llama32_3b")
    parser.add_argument("--mode", choices=("inspect", "prepare", "run", "report"), required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from klrd_rs.portable.profile import ModelProfile

    profile = ModelProfile.load(args.profile)
    root = Path(__file__).resolve().parent
    output = args.output or profile.output_path(root, "e1", "run_001")
    if args.mode == "inspect":
        print(json.dumps({
            "mode": "inspect",
            "profile_id": profile.profile_id,
            "model_id": profile.model_id,
            "revision": profile.revision,
            "output": str(output),
            "status": "OFFLINE_PROFILE_VALID",
        }, indent=2))
        return 0
    from klrd_rs.portable.e1 import run_e1

    result = run_e1(profile, output, args.mode)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
