#!/usr/bin/env python3
"""Convert solved HyperContact-3D CalculiX outputs to ValveCase directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from valgraphnet.calculix_results import convert_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--allow-nodal-stress-fallback",
        action="store_true",
        help="allow cell stress averaged from extrapolated FRD nodal stress when DAT is absent",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_benchmark(
        args.manifest,
        args.output,
        splits=args.split or None,
        case_ids=args.case_id or None,
        require_dat_stress=not args.allow_nodal_stress_fallback,
        force=args.force,
    )
    print(json.dumps(summary["split_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
