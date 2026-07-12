"""Run the CUDA-only frozen-potential force identifiability diagnostic."""

from __future__ import annotations

import argparse
import json

from valgraphnet.config import load_config
from valgraphnet.force_identifiability import run_force_identifiability


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one positive inverse-inertia scale on train-only internal "
            "forces, freeze it, then evaluate fixed even-val20 once."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/deforming_plate_force_identifiability.yaml",
    )
    args = parser.parse_args()
    artifacts = run_force_identifiability(
        load_config(args.config), config_path=args.config
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
