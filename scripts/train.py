from __future__ import annotations

import argparse

from valgraphnet.config import load_config
from valgraphnet.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ValGraphNet.")
    parser.add_argument("--config", default="configs/valve_hybrid.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    best_path = run_training(cfg)
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

