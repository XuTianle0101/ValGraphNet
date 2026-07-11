"""Generate a deterministic geometry/material-disjoint Valve split.

Each exported case must contain ``metadata.json`` with a ``case_id`` matching
its directory name and explicit geometry/material identifiers.  No ODB values
are synthesized or inferred by this command.

Example:

    python scripts/generate_valve_ood_splits.py \
      --data-root data/valve_chp_cases \
      --output data/valve_chp_cases/splits.strict_ood.json
"""

from __future__ import annotations

import argparse
import json

from valgraphnet.data.valve_ood import write_strict_ood_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing exported case directories.",
    )
    parser.add_argument("--output", required=True, help="Destination JSON split file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--geometry-key",
        default="geometry_id",
        help="Dotted key in each metadata.json identifying reference geometry.",
    )
    parser.add_argument(
        "--material-key",
        default="material_id",
        help="Dotted key in each metadata.json identifying the material parameter set.",
    )
    args = parser.parse_args()
    payload = write_strict_ood_split(
        args.data_root,
        args.output,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        geometry_key=args.geometry_key,
        material_key=args.material_key,
    )
    print(json.dumps(payload["audit"], indent=2, sort_keys=True))
    print(f"Wrote strict Valve OOD split: {args.output}")


if __name__ == "__main__":
    main()
