#!/usr/bin/env python3
"""Run generated HyperContact-3D decks with CalculiX.

Example
-------
python scripts/run_hypercontact3d.py \
    --manifest data/hypercontact3d_raw/manifest.json \
    --ccx C:/CalculiX/ccx.exe --split train --workers 2 --solver-threads 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from valgraphnet.hypercontact_solver import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ccx", default="ccx", help="CalculiX executable path or command name")
    parser.add_argument(
        "--ccx-arg",
        action="append",
        default=[],
        help="argument inserted before '-i model'; repeat for command wrappers",
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="rerun cases with valid outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        args.manifest,
        [args.ccx, *args.ccx_arg],
        splits=args.split or None,
        case_ids=args.case_id or None,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        force=args.force,
        solver_threads=args.solver_threads,
        summary_path=args.summary,
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    failed = sum(
        count
        for status, count in summary["counts"].items()
        if status not in {"succeeded", "skipped"}
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
