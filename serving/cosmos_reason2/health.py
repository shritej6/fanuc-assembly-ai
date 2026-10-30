"""Health check for the Cosmos Reason 2 8B build-check endpoint: right model, right weights, parseable answers.

    uv run python -m serving.cosmos_reason2.health [--base-url http://127.0.0.1:8000/v1]

Passes (exit 0) only if ``/v1/models`` serves exactly ``Cosmos-Reason2-8B`` from ``nvidia/Cosmos-Reason2-8B`` and
both bundled rendered set D crops (a seated and a raised top part, sim only) come back from the production port
(``CosmosReason2Endpoint.check``) as a parsed verdict. The verdicts and latencies are printed; a wrong verdict does not
fail the check (accuracy is measured by eval/cosmos_reason2_bench.py), an unparseable or missing answer does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.cosmos_reason import CosmosReasonUnavailable, FinalBuildRequest, InstalledInstanceLineage
from agent.cosmos_reason2_endpoint import (
    MODEL_ID,
    MODEL_REPO,
    CosmosReason2Endpoint,
    DirectoryImageSource,
    EndpointConfig,
)
from assembly.catalog import load_catalog

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _served_model(base_url: str) -> tuple[str, str]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        models = json.load(response)["data"]
    if len(models) != 1:
        raise RuntimeError(f"expected one served model, got {[m.get('id') for m in models]}")
    return models[0]["id"], models[0].get("root", "")


def _request(case: dict, image: bytes) -> FinalBuildRequest:
    steps = load_catalog().product(case["product"]).steps[: case["step"]]
    now = datetime.now(UTC)
    return FinalBuildRequest(
        "health", 0, "health", f"health-{case['capture_id']}", case["product"],
        tuple(InstalledInstanceLineage(f"health-{s.step}", s.object_id) for s in steps),
        case["capture_id"], hashlib.sha256(image).hexdigest(), now, now + timedelta(minutes=5), "health-fixture",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    served, root = _served_model(args.base_url)
    if served != MODEL_ID or root != MODEL_REPO:
        print(f"FAIL: endpoint serves {served!r} from {root!r}, expected {MODEL_ID!r} from {MODEL_REPO!r}")
        return 1
    print(f"model ok: {served} from {root}")
    endpoint = CosmosReason2Endpoint(
        EndpointConfig(base_url=args.base_url), DirectoryImageSource(FIXTURES), load_catalog()
    )
    cases = json.loads((FIXTURES / "cases.json").read_text())
    ok = True
    for case in cases:
        image = (FIXTURES / f"{case['capture_id']}.jpg").read_bytes()
        started = time.perf_counter()
        try:
            result = endpoint.check(_request(case, image))
            verdict = "pass" if result.passed else "fail"
            print(f"{case['capture_id']} ({case['state']}): {verdict} - {result.explanation}", end="")
        except CosmosReasonUnavailable as error:
            ok = False
            print(f"{case['capture_id']} ({case['state']}): UNAVAILABLE - {error}", end="")
        print(f" [{time.perf_counter() - started:.1f} s]")
    print("HEALTHY" if ok else "FAIL: an answer did not parse")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
