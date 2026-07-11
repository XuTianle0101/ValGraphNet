"""Run generated HyperContact-3D CalculiX decks reproducibly.

The runner intentionally invokes the solver without a shell, isolates every
case in its generated directory, and records machine-readable status.  It is
usable from tests with a command prefix such as ``[python, fake_ccx.py]`` and
from the command line with a regular ``ccx`` executable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CaseRunResult:
    """Outcome of one solver invocation."""

    case_id: str
    split: str
    status: str
    returncode: int | None
    duration_seconds: float
    message: str
    outputs: dict[str, dict[str, int | float]]
    convergence: dict[str, Any]


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _safe_manifest_path(root: Path, relative: str, *, description: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{description} escapes benchmark root: {relative!r}") from exc
    return candidate


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load and minimally validate a generated benchmark manifest."""

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError(f"{manifest_path}: manifest must contain a cases list")
    case_ids: set[str] = set()
    for entry in manifest["cases"]:
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest_path}: every case entry must be an object")
        case_id = str(entry.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(f"{manifest_path}: missing or duplicate case id {case_id!r}")
        case_ids.add(case_id)
        if not isinstance(entry.get("deck"), str):
            raise ValueError(f"{manifest_path}: {case_id} has no deck path")
    return manifest_path.parent, manifest


def select_cases(
    manifest: Mapping[str, Any],
    *,
    splits: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic manifest entries by split and/or case id."""

    split_filter = set(splits or ())
    id_filter = set(case_ids or ())
    known_ids = {str(entry["case_id"]) for entry in manifest["cases"]}
    missing = sorted(id_filter - known_ids)
    if missing:
        raise KeyError(f"unknown HyperContact case ids: {missing}")
    selected = [
        dict(entry)
        for entry in manifest["cases"]
        if (not split_filter or str(entry.get("split")) in split_filter)
        and (not id_filter or str(entry["case_id"]) in id_filter)
    ]
    return sorted(selected, key=lambda entry: str(entry["case_id"]))


def resolve_solver_command(command: Sequence[str]) -> list[str]:
    """Resolve the executable while retaining any explicit command prefix."""

    if not command:
        raise ValueError("solver command must not be empty")
    executable = os.path.expandvars(os.path.expanduser(str(command[0])))
    resolved = shutil.which(executable)
    if resolved is None and Path(executable).is_file():
        resolved = str(Path(executable).resolve())
    if resolved is None:
        raise FileNotFoundError(
            f"CalculiX executable not found: {command[0]!r}; pass --ccx with an explicit path"
        )
    return [resolved, *(str(argument) for argument in command[1:])]


def _output_state(paths: Iterable[Path]) -> tuple[dict[str, dict[str, int | float]], list[str]]:
    state: dict[str, dict[str, int | float]] = {}
    missing: list[str] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(str(path))
            continue
        stat = path.stat()
        state[path.name] = {
            "size_bytes": int(stat.st_size),
            "modified_unix_seconds": float(stat.st_mtime),
        }
    return state, missing


def _convergence_evidence(
    sta_path: Path,
    stdout: bytes,
    *,
    target_step_time: float,
) -> dict[str, Any]:
    last_row: list[float] | None = None
    if sta_path.is_file():
        for line in sta_path.read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = _NUMBER.findall(line)
            if len(tokens) < 7:
                continue
            try:
                values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
            except ValueError:
                continue
            if all(float(values[index]).is_integer() for index in range(4)):
                last_row = values
    reached_target = False
    if last_row is not None:
        tolerance = max(1.0e-8, 1.0e-5 * abs(float(target_step_time)))
        reached_target = abs(last_row[5] - float(target_step_time)) <= tolerance
    stdout_finished = b"job finished" in stdout.lower()
    return {
        "sta_last_increment": int(last_row[1]) if last_row is not None else None,
        "sta_last_step": int(last_row[0]) if last_row is not None else None,
        "sta_step_time": float(last_row[5]) if last_row is not None else None,
        "sta_total_time": float(last_row[4]) if last_row is not None else None,
        "sta_reached_target": bool(reached_target),
        "stdout_job_finished": bool(stdout_finished),
        "target_step_time": float(target_step_time),
        "validated": bool(reached_target or stdout_finished),
    }


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


def run_case(
    benchmark_root: str | Path,
    entry: Mapping[str, Any],
    solver_command: Sequence[str],
    *,
    timeout_seconds: float | None = None,
    force: bool = False,
    solver_threads: int = 1,
) -> CaseRunResult:
    """Run one manifest case and validate the declared solver outputs."""

    root = Path(benchmark_root).resolve()
    case_id = str(entry["case_id"])
    split = str(entry.get("split", "unknown"))
    deck = _safe_manifest_path(root, str(entry["deck"]), description=f"{case_id} deck")
    if not deck.is_file():
        raise FileNotFoundError(f"missing input deck for {case_id}: {deck}")
    expected_hash = entry.get("deck_sha256")
    if expected_hash and _sha256(deck) != str(expected_hash):
        raise ValueError(f"input deck hash mismatch for {case_id}: {deck}")
    expected = [
        _safe_manifest_path(root, str(relative), description=f"{case_id} output")
        for relative in entry.get("expected_outputs", ())
    ]
    if not expected:
        expected = [deck.with_suffix(suffix) for suffix in (".frd", ".dat", ".sta")]
    status_path = deck.parent / "solver_status.json"
    stdout_path = deck.parent / "solver.stdout.log"
    stderr_path = deck.parent / "solver.stderr.log"
    lock_path = deck.parent / ".solver.lock"

    existing_outputs, missing = _output_state(expected)
    if not force and not missing and status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
        if previous.get("status") == "succeeded":
            convergence = dict(previous.get("convergence", {}))
            if not convergence.get("validated", False):
                target_step_time = float(
                    dict(entry.get("derived", {})).get("step_duration", 1.0)
                )
                saved_stdout = stdout_path.read_bytes() if stdout_path.is_file() else b""
                sta_path = next(
                    (path for path in expected if path.suffix.lower() == ".sta"),
                    deck.with_suffix(".sta"),
                )
                convergence = _convergence_evidence(
                    sta_path,
                    saved_stdout,
                    target_step_time=target_step_time,
                )
            if convergence.get("validated", False):
                return CaseRunResult(
                    case_id=case_id,
                    split=split,
                    status="skipped",
                    returncode=previous.get("returncode"),
                    duration_seconds=0.0,
                    message="validated existing successful outputs",
                    outputs=existing_outputs,
                    convergence=convergence,
                )

    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return CaseRunResult(
            case_id=case_id,
            split=split,
            status="locked",
            returncode=None,
            duration_seconds=0.0,
            message=f"case is locked by another runner: {lock_path}",
            outputs=existing_outputs,
            convergence={},
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
        lock.write(f"pid={os.getpid()} started={_utc_now()}\n")

    started_at = _utc_now()
    started = time.monotonic()
    command = [*solver_command, "-i", deck.stem]
    environment = os.environ.copy()
    threads = max(1, int(solver_threads))
    environment.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "CCX_NPROC_RESULTS": str(threads),
            "CCX_NPROC_EQUATION_SOLVER": str(threads),
        }
    )
    _atomic_json(
        status_path,
        {
            "case_id": case_id,
            "command": command,
            "deck_sha256": _sha256(deck),
            "solver_threads": threads,
            "split": split,
            "started_at": started_at,
            "status": "running",
        },
    )

    status = "failed"
    message = "solver did not complete"
    returncode: int | None = None
    stdout = b""
    stderr = b""
    convergence: dict[str, Any] = {}
    try:
        process = subprocess.Popen(
            command,
            cwd=deck.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name != "nt",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.output or b""
            stderr = exc.stderr or b""
            _terminate_process_tree(process)
            trailing_stdout, trailing_stderr = process.communicate()
            stdout += trailing_stdout or b""
            stderr += trailing_stderr or b""
            returncode = process.returncode
            status = "timed_out"
            message = f"solver exceeded timeout of {timeout_seconds:g} seconds"

        output_state, missing = _output_state(expected)
        sta_path = next(
            (path for path in expected if path.suffix.lower() == ".sta"),
            deck.with_suffix(".sta"),
        )
        target_step_time = float(dict(entry.get("derived", {})).get("step_duration", 1.0))
        convergence = _convergence_evidence(
            sta_path,
            stdout,
            target_step_time=target_step_time,
        )
        if status != "timed_out":
            if returncode != 0:
                message = f"solver exited with return code {returncode}"
            elif missing:
                message = (
                    "solver exited successfully but expected outputs are missing or empty: "
                    + ", ".join(missing)
                )
            elif not convergence["validated"]:
                message = (
                    "solver exited successfully but convergence evidence did not "
                    f"reach step time {target_step_time:g}"
                )
            else:
                status = "succeeded"
                message = "solver completed and all declared outputs are non-empty"
    except OSError as exc:
        output_state, _ = _output_state(expected)
        message = f"could not start solver: {exc}"
    finally:
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        duration = time.monotonic() - started
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    result = CaseRunResult(
        case_id=case_id,
        split=split,
        status=status,
        returncode=returncode,
        duration_seconds=float(duration),
        message=message,
        outputs=output_state,
        convergence=convergence,
    )
    _atomic_json(
        status_path,
        {
            **asdict(result),
            "command": command,
            "deck_sha256": _sha256(deck),
            "finished_at": _utc_now(),
            "solver_threads": threads,
            "started_at": started_at,
        },
    )
    return result


def run_benchmark(
    manifest_path: str | Path,
    solver_command: Sequence[str],
    *,
    splits: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    workers: int = 1,
    timeout_seconds: float | None = None,
    force: bool = False,
    solver_threads: int = 1,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run selected cases concurrently and write a deterministic batch summary."""

    root, manifest = load_manifest(manifest_path)
    command = resolve_solver_command(solver_command)
    entries = select_cases(manifest, splits=splits, case_ids=case_ids)
    if not entries:
        raise ValueError("case selection is empty")
    worker_count = max(1, int(workers))
    results: list[CaseRunResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_case,
                root,
                entry,
                command,
                timeout_seconds=timeout_seconds,
                force=force,
                solver_threads=solver_threads,
            ): str(entry["case_id"])
            for entry in entries
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                entry = next(item for item in entries if str(item["case_id"]) == futures[future])
                results.append(
                    CaseRunResult(
                        case_id=futures[future],
                        split=str(entry.get("split", "unknown")),
                        status="failed",
                        returncode=None,
                        duration_seconds=0.0,
                        message=str(exc),
                        outputs={},
                        convergence={},
                    )
                )
    results.sort(key=lambda result: result.case_id)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = {
        "case_count": len(results),
        "counts": dict(sorted(counts.items())),
        "finished_at": _utc_now(),
        "manifest": str(Path(manifest_path).resolve()),
        "results": [asdict(result) for result in results],
        "solver_command": command,
        "solver_threads_per_case": max(1, int(solver_threads)),
        "workers": worker_count,
    }
    destination = Path(summary_path) if summary_path else root / "solver_run_summary.json"
    _atomic_json(destination.resolve(), summary)
    return summary
