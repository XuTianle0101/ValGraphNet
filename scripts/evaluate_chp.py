from __future__ import annotations

import argparse
import json

from valgraphnet.chp_rollout import run_chp_rollouts
from valgraphnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run schema-v2 CHP-GNS rollouts and physical evaluation on CUDA."
    )
    parser.add_argument(
        "--config", default="configs/deforming_plate_chp.full400.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    result = run_chp_rollouts(
        load_config(args.config),
        args.checkpoint,
        args.out,
        split=args.split,
        max_cases=args.max_cases,
    )
    print(json.dumps(result["metrics"]["summary"], indent=2))


if __name__ == "__main__":
    main()
