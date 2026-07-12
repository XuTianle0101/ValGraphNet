from __future__ import annotations

import argparse
import json

from valgraphnet.config import load_config
from valgraphnet.legacy_rollout_export import export_legacy_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export existing ValGraphNet rollouts in the shared format."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--case-selection",
        choices=("head", "even"),
        default=None,
    )
    args = parser.parse_args()
    result = export_legacy_rollouts(
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
