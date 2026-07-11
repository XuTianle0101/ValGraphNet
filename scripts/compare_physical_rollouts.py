from __future__ import annotations

import argparse
import json
from pathlib import Path

from valgraphnet.physical_evaluation import (
    compare_experiments,
    evaluate_prediction_directory,
    save_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standardized simulator rollout directories."
    )
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for native, repo, fair_mgn, and chp_gns.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    experiments = {}
    for specification in args.experiment:
        name, separator, directory = specification.partition("=")
        if not separator:
            raise ValueError("--experiment must use NAME=PATH")
        experiments[name] = evaluate_prediction_directory(
            args.case_root,
            args.split_file,
            args.split,
            directory,
        )
    comparison = compare_experiments(
        experiments,
        baseline=args.baseline,
        candidate=args.candidate,
        bootstrap_samples=args.bootstrap_samples,
    )
    save_comparison(Path(args.out), comparison)
    print(json.dumps(comparison["acceptance"], indent=2))


if __name__ == "__main__":
    main()
