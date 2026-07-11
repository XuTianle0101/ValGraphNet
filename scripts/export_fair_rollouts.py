from __future__ import annotations

import argparse
import json

from valgraphnet.config import load_config
from valgraphnet.fair_rollout_export import export_fair_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export corrected fair-MGN rollouts in the shared format."
    )
    parser.add_argument(
        "--config", default="configs/deforming_plate_fair_mgn.full400.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    result = export_fair_rollouts(
        load_config(args.config),
        args.checkpoint,
        args.out,
        split=args.split,
        max_cases=args.max_cases,
    )
    print(json.dumps(result["metrics"]["summary"], indent=2))


if __name__ == "__main__":
    main()
