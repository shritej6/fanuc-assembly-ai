"""Run the remote PatchCore production job with an exact-ID Brev stop guard.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perception.brev.t14_benchmark_gate import UTC_FORMAT

INSTANCE_NAME = "fa-t14"


class RuntimeGuardError(RuntimeError):
    """Raised when guarded production cannot remain inside its paid-runtime boundary."""


ListInstances = Callable[[], Any]
RemoteRun = Callable[[str, str, str, float], None]
StopInstance = Callable[[str], None]


def _now() -> datetime:
    return datetime.now(UTC)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeGuardError(f"benchmark gate must be a real file: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeGuardError("benchmark gate must be a JSON object")
    return payload


def _records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if any(key in value for key in ("id", "instance_id", "instanceId")):
            records.append(value)
        for child in value.values():
            records.extend(_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_records(child))
    return records


def _field(record: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _verify_live_instance(payload: Any, instance_name: str, instance_id: str) -> None:
    matches = []
    for record in _records(payload):
        name = _field(record, ("name", "instance_name", "instanceName"))
        identifier = _field(record, ("id", "instance_id", "instanceId"))
        if name == instance_name:
            matches.append(identifier)
    if matches != [instance_id]:
        raise RuntimeGuardError("live Brev name-to-ID mapping does not match the guarded instance")


def _remaining_seconds(gate_path: Path, now: datetime) -> float:
    gate = _json(gate_path)
    try:
        deadline = datetime.strptime(gate["budget_deadline_utc"], UTC_FORMAT).replace(tzinfo=UTC)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeGuardError("benchmark gate lacks a valid runtime deadline") from exc
    remaining = (deadline - now).total_seconds()
    if remaining <= 0:
        raise RuntimeGuardError("benchmark gate runtime budget is expired")
    return remaining


def run_guarded(
    *,
    instance_name: str,
    instance_id: str,
    gate_path: Path,
    remote_env: str,
    remote_code_root: str,
    list_instances: ListInstances,
    run_remote: RemoteRun,
    stop_instance: StopInstance,
    now: datetime | None = None,
) -> None:
    """Run production on the dedicated instance and stop it when time runs out or the run fails."""
    if instance_name != INSTANCE_NAME or re.fullmatch(r"[a-z0-9]+", instance_id) is None:
        raise RuntimeGuardError("guard is restricted to the dedicated fa-t14 Brev instance")
    _verify_live_instance(list_instances(), instance_name, instance_id)
    try:
        remaining = _remaining_seconds(gate_path, now or _now())
        run_remote(instance_id, remote_env, remote_code_root, remaining)
    except (Exception, KeyboardInterrupt) as exc:
        try:
            stop_instance(instance_id)
        except Exception as stop_exc:
            raise RuntimeGuardError("production failed and exact-ID Brev stop also failed") from stop_exc
        raise RuntimeGuardError("production failed; the exact Brev instance was stopped") from exc


def _list_instances() -> Any:
    result = subprocess.run(["brev", "ls", "--json"], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _run_remote(instance_id: str, remote_env: str, remote_code_root: str, timeout: float) -> None:
    script = (
        'set -a; source "$1"; set +a; export T14_GUARDED_ORCHESTRATION=1; '
        'exec "$2/perception/brev/t14_run_all_patchcore.sh"'
    )
    command = shlex.join(["bash", "-c", script, "t14", remote_env, remote_code_root])
    subprocess.run(
        ["brev", "exec", instance_id, command],
        check=True,
        timeout=timeout,
    )


def _stop_instance(instance_id: str) -> None:
    subprocess.run(["brev", "stop", instance_id], check=True)


def main() -> None:
    """Parse arguments and run the guarded production job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-name", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--benchmark-gate", type=Path, required=True)
    parser.add_argument("--remote-env", required=True)
    parser.add_argument("--remote-code-root", required=True)
    args = parser.parse_args()
    run_guarded(
        instance_name=args.instance_name,
        instance_id=args.instance_id,
        gate_path=args.benchmark_gate,
        remote_env=args.remote_env,
        remote_code_root=args.remote_code_root,
        list_instances=_list_instances,
        run_remote=_run_remote,
        stop_instance=_stop_instance,
    )


if __name__ == "__main__":
    main()
