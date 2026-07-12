"""Export two-level MultiScale MGN rollouts in the shared artifact format."""

from __future__ import annotations

import argparse
import json

from valgraphnet.config import load_config
from valgraphnet.multiscale_rollout_export import export_multiscale_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/deforming_plate_multiscale_mgn.full400.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-selection", choices=("head", "even"), default="head")
    args = parser.parse_args()
    result = export_multiscale_rollouts(
        load_config(args.config),
        args.checkpoint,
        args.out,
        split=args.split,
        max_cases=args.max_cases,
        case_selection=args.case_selection,
    )
    print(json.dumps(result["metrics"]["summary"], indent=2))


if __name__ == "__main__":
    main()
