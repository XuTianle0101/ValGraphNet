from __future__ import annotations

import argparse

from valgraphnet.config import load_config
from valgraphnet.rollout import run_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ValGraphNet autoregressive rollout.")
    parser.add_argument("--config", default="configs/valve_hybrid.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--case", required=True, help="Exported case directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--steps", type=int, default=None, help="Optional number of rollout steps.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_rollout(cfg, args.checkpoint, args.case, args.out, steps=args.steps)
    print(f"rollout written to: {out}")


if __name__ == "__main__":
    main()

