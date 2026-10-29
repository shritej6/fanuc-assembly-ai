import base64
import hashlib
import json
import threading
import urllib.error
import urllib.request
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assembly.state_machine import PackMLState
from industry.dashboard import DashboardConfigurationError, config_from_args, create_server
from industry.dashboard_assets import DASHBOARD_HTML
from industry.oee import (
    DashboardSchedule,
    EventSource,
    PackMLEvent,
    compute_dashboard_snapshot,
)

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)


def _snapshot() -> object:
    schedule = DashboardSchedule(NOW, NOW.replace(minute=1), {})
    events = (PackMLEvent(NOW, EventSource.MOCK, PackMLState.EXECUTE, None, 1),)
    return compute_dashboard_snapshot(events, schedule, now=NOW, stale_after_s=10)


def _request(server: object, path: str) -> tuple[int, dict[str, str], bytes]:
    port = server.server_address[1]  # type: ignore[attr-defined]
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        response = urllib.request.urlopen(request, timeout=2)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
    return response.status, dict(response.headers), response.read()


def _serve(server: object) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)  # type: ignore[attr-defined]
    thread.start()
    return thread


def test_serves_interface_and_snapshot_read_only() -> None:
    server = create_server("127.0.0.1", 0, _snapshot)  # type: ignore[arg-type]
    thread = _serve(server)
    try:
        html_status, html_headers, html = _request(server, "/")
        api_status, api_headers, api = _request(server, "/api/snapshot")
        missing_status, _, missing = _request(server, "/missing")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert html_status == 200
    assert b"Assembly cell status" in html
    assert "no-store" in html_headers["Cache-Control"]
    assert "default-src 'none'" in html_headers["Content-Security-Policy"]
    assert "unsafe-inline" not in html_headers["Content-Security-Policy"]
    assert "sha256-" in html_headers["Content-Security-Policy"]
    assert api_status == 200
    assert api_headers["Content-Type"].startswith("application/json")
    assert json.loads(api)["source"] == "mock"
    assert missing_status == 404
    assert json.loads(missing) == {"error": "not found"}


def test_content_security_policy_hashes_match_the_served_inline_assets() -> None:
    server = create_server("127.0.0.1", 0, _snapshot)  # type: ignore[arg-type]
    thread = _serve(server)
    try:
        _, headers, html = _request(server, "/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    document = html.decode()
    policy = headers["Content-Security-Policy"]
    for tag in ("style", "script"):
        content = document.partition(f"<{tag}>")[2].partition(f"</{tag}>")[0]
        digest = base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in policy


def test_provider_failure_returns_fail_closed_red_response() -> None:
    def fail() -> object:
        raise ValueError("malformed run log")

    server = create_server("127.0.0.1", 0, fail)  # type: ignore[arg-type]
    thread = _serve(server)
    try:
        status, _, body = _request(server, "/api/snapshot")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 503
    assert json.loads(body) == {"error": "malformed run log", "rag_status": "red"}


def test_dashboard_uses_safe_text_rendering() -> None:
    assert "innerHTML" not in DASHBOARD_HTML
    assert "textContent" in DASHBOARD_HTML
    assert "Data source" in DASHBOARD_HTML
    assert "prefers-reduced-motion" in DASHBOARD_HTML
    assert 'id="status-announcement"' in DASHBOARD_HTML
    assert 'class="status-strip" aria-live' not in DASHBOARD_HTML


def test_cli_requires_timezone_and_unique_measured_ideal_cycles(tmp_path: Path) -> None:
    args = Namespace(
        log=tmp_path / "run.jsonl",
        planned_start="2026-10-29T15:00:00Z",
        planned_end="2026-10-29T16:00:00Z",
        ideal_cycle=["1=42.5", "2=55"],
        stale_after_s=10.0,
    )

    config = config_from_args(args)

    assert config.schedule.ideal_cycle_time_s == {1: 42.5, 2: 55.0}


def test_cli_rejects_duplicate_ideal_cycle() -> None:
    args = Namespace(
        log=Path("run.jsonl"),
        planned_start="2026-10-29T15:00:00Z",
        planned_end="2026-10-29T16:00:00Z",
        ideal_cycle=["1=42", "1=43"],
        stale_after_s=10.0,
    )

    with pytest.raises(DashboardConfigurationError, match="duplicate"):
        config_from_args(args)


def test_server_rejects_non_loopback_binding() -> None:
    with pytest.raises(DashboardConfigurationError, match="loopback"):
        create_server("0.0.0.0", 8080, _snapshot)  # type: ignore[arg-type]


def test_cli_rejects_nonpositive_staleness_policy() -> None:
    args = Namespace(
        log=Path("run.jsonl"),
        planned_start="2026-10-29T15:00:00Z",
        planned_end="2026-10-29T16:00:00Z",
        ideal_cycle=[],
        stale_after_s=0.0,
    )

    with pytest.raises(DashboardConfigurationError, match="positive finite"):
        config_from_args(args)
