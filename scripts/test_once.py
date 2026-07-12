"""Freeze, verify, and consume a formal test-once experiment protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from valgraphnet.test_once import (
    TestOnceError,
    claim_test_run,
    freeze_experiment,
    protocol_paths,
    run_locked_test_plan,
    verify_frozen_experiment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lock validation-selected checkpoints before a single guarded test run. "
            "Freeze/verify never read test trajectories."
        )
    )
    parser.add_argument(
        "--registry",
        default="outputs/test_once_registry",
        help="Persistent registry; changing it creates a separate governance domain.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    freeze = subparsers.add_parser("freeze", help="Create an immutable experiment lock.")
    freeze.add_argument("--spec", required=True, help="JSON/YAML frozen experiment spec.")
    freeze.add_argument("--workspace", default=".")

    for action in ("verify", "claim", "run", "status"):
        command = subparsers.add_parser(action)
        command.add_argument("--experiment-id", required=True)
    claim = subparsers.choices["claim"]
    claim.add_argument(
        "--consume-test-attempt",
        action="store_true",
        help="Required safeguard for an external orchestrator claim.",
    )
    run = subparsers.choices["run"]
    run.add_argument(
        "--execute-frozen-test-plan",
        action="store_true",
        help="Required safeguard: atomically consume the only attempt and execute it.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    registry = Path(args.registry)
    try:
        if args.action == "freeze":
            path = freeze_experiment(
                args.spec,
                registry,
                workspace_root=args.workspace,
            )
            print(f"frozen: {path}")
            return
        if args.action == "verify":
            manifest = verify_frozen_experiment(registry, args.experiment_id)
            print(
                json.dumps(
                    {
                        "experiment_id": manifest["experiment_id"],
                        "models": [model["name"] for model in manifest["models"]],
                        "state": "verified_not_claimed",
                        "test_data_accessed": False,
                    },
                    sort_keys=True,
                )
            )
            return
        if args.action == "claim":
            if not args.consume_test_attempt:
                raise TestOnceError(
                    "claim requires --consume-test-attempt; this is irreversible"
                )
            path = claim_test_run(registry, args.experiment_id)
            print(f"claimed (no retry): {path}")
            return
        if args.action == "run":
            if not args.execute_frozen_test_plan:
                raise TestOnceError(
                    "run requires --execute-frozen-test-plan; this consumes the only attempt"
                )
            path = run_locked_test_plan(registry, args.experiment_id)
            print(f"completed: {path}")
            return
        manifest_path, claim_path, result_path = protocol_paths(
            registry, args.experiment_id
        )
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "frozen": manifest_path.exists(),
                    "claim": str(claim_path),
                    "claimed": claim_path.exists(),
                    "result": str(result_path),
                    "finished": result_path.exists(),
                },
                sort_keys=True,
            )
        )
    except TestOnceError as error:
        raise SystemExit(f"test-once protocol rejected: {error}") from error


if __name__ == "__main__":
    main()
