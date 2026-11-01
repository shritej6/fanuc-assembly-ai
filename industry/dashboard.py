"""Read-only HTTP dashboard for evidence-backed OEE and Andon status.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from industry.dashboard_assets import DASHBOARD_HTML
from industry.oee import DashboardSchedule, DashboardSnapshot, compute_dashboard_snapshot
from industry.run_log import load_run_log

LOGGER = logging.getLogger(__name__)


def _inline_hash(tag: str) -> str:
    opening, closing = f"<{tag}>", f"</{tag}>"
    content = DASHBOARD_HTML.partition(opening)[2].partition(closing)[0]
    digest = hashlib.sha256(content.encode()).digest()
    return "sha256-" + base64.b64encode(digest).decode()


_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    f"script-src '{_inline_hash('script')}'; "
    f"style-src '{_inline_hash('style')}'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'"
)


class DashboardConfigurationError(ValueError):
    """CLI configuration cannot produce an evidence-backed dashboard."""


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    log_path: Path
    schedule: DashboardSchedule
    stale_after_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.log_path, Path):
            raise DashboardConfigurationError("log_path must be a pathlib.Path")
        if isinstance(self.stale_after_s, bool) or not isinstance(self.stale_after_s, (int, float)):
            raise DashboardConfigurationError("stale_after_s must be a positive finite number")
        if not math.isfinite(self.stale_after_s) or self.stale_after_s <= 0.0:
            raise DashboardConfigurationError("stale_after_s must be a positive finite number")


SnapshotProvider = Callable[[], DashboardSnapshot]


def snapshot_from_log(config: DashboardConfig, *, clock: Callable[[], datetime] | None = None) -> DashboardSnapshot:
    now = (clock or _utc_now)()
    events = load_run_log(config.log_path)
    return compute_dashboard_snapshot(events, config.schedule, now=now, stale_after_s=config.stale_after_s)


def create_server(host: str, port: int, provider: SnapshotProvider) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise DashboardConfigurationError("dashboard host must be loopback (127.0.0.1 or localhost)")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise DashboardConfigurationError("port must be an integer from 0 through 65535")
    if not callable(provider):
        raise DashboardConfigurationError("snapshot provider must be callable")
    return ThreadingHTTPServer((host, port), _handler(provider))


def _handler(provider: SnapshotProvider) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.partition("?")[0]
            if path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", DASHBOARD_HTML.encode())
                return
            if path == "/api/snapshot":
                self._snapshot()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _snapshot(self) -> None:
            try:
                snapshot = provider()
            except Exception as error:  # HTTP boundary converts typed domain errors to a fail-closed response.
                LOGGER.exception("dashboard snapshot failed")
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error), "rag_status": "red"})
                return
            self._send_json(HTTPStatus.OK, snapshot.as_dict())

        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self._send(status, "application/json; charset=utf-8", body)

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("dashboard request: " + format, *args)

    return DashboardHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the read-only FANUC OEE and Andon dashboard")
    parser.add_argument("--log", type=Path, required=True, help="append-only JSONL run log")
    parser.add_argument("--planned-start", required=True, help="planned window start, ISO 8601 with timezone")
    parser.add_argument("--planned-end", required=True, help="planned window end, ISO 8601 with timezone")
    parser.add_argument(
        "--ideal-cycle",
        action="append",
        default=[],
        metavar="PRODUCT=SECONDS",
        help="measured ideal cycle time; repeat for each product in the log",
    )
    parser.add_argument("--stale-after-s", type=float, required=True, help="freshness policy in seconds")
    parser.add_argument("--host", default="127.0.0.1", help="listen address; defaults to loopback")
    parser.add_argument("--port", type=int, default=8080, help="listen port")
    return parser


def config_from_args(args: argparse.Namespace) -> DashboardConfig:
    schedule = DashboardSchedule(
        _timestamp(args.planned_start, "planned-start"),
        _timestamp(args.planned_end, "planned-end"),
        _ideal_cycles(args.ideal_cycle),
    )
    return DashboardConfig(args.log, schedule, args.stale_after_s)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        server = create_server(args.host, args.port, lambda: snapshot_from_log(config))
    except (ValueError, OSError) as error:
        LOGGER.error("cannot start dashboard: %s", error)
        return 2
    LOGGER.info("dashboard listening on http://%s:%d", *server.server_address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("dashboard stopped")
    finally:
        server.server_close()
    return 0


def _ideal_cycles(values: Sequence[str]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        try:
            product_text, seconds_text = value.split("=", maxsplit=1)
            product, seconds = int(product_text), float(seconds_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise DashboardConfigurationError(f"invalid --ideal-cycle {value!r}; expected PRODUCT=SECONDS") from error
        if product in result:
            raise DashboardConfigurationError(f"duplicate ideal cycle time for product {product}")
        result[product] = seconds
    return result


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise DashboardConfigurationError(f"{field} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise DashboardConfigurationError(f"{field} must be an ISO 8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DashboardConfigurationError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
