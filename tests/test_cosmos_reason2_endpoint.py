import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from agent.cosmos_reason import (
    CosmosEvidenceSource,
    CosmosReasonUnavailable,
    FinalBuildRequest,
    InstalledInstanceLineage,
)
from agent.cosmos_reason2_endpoint import (
    MODEL_ID,
    CosmosReason2Endpoint,
    EndpointConfig,
    parse_completion,
)
from assembly.catalog import load_catalog

NOW = datetime(2026, 10, 30, 15, tzinfo=UTC)
JPEG = b"\xff\xd8\xff\xe0" + b"rendered-sim-image"


class Images:
    def __init__(self, data: bytes = JPEG) -> None:
        self.data = data

    def image_bytes(self, capture_id: str) -> bytes:
        if capture_id != "capture-1":
            raise KeyError(capture_id)
        return self.data


def _request(**changes: object) -> FinalBuildRequest:
    recipe = load_catalog().product(2)
    values = {
        "run_id": "run-1",
        "rollout": 0,
        "task_id": "task-1",
        "request_id": "request-1",
        "product": 2,
        "installed_instances": tuple(
            InstalledInstanceLineage(f"physical-{step.step}", step.object_id) for step in recipe.steps
        ),
        "image_capture_id": "capture-1",
        "image_sha256": hashlib.sha256(JPEG).hexdigest(),
        "observed_at": NOW,
        "expires_at": NOW + timedelta(minutes=1),
        "evidence_source": "trusted-test-provider",
    }
    values.update(changes)
    return FinalBuildRequest(**values)  # type: ignore[arg-type]


def _completion(content: str, model: str = MODEL_ID, finish: str = "stop", reasoning: str = "looked") -> bytes:
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    return json.dumps({"model": model, "choices": [{"message": message, "finish_reason": finish}]}).encode()


class Transport:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, payload: bytes, timeout_s: float) -> bytes:
        self.calls.append((url, json.loads(payload)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]


def _endpoint(transport: Transport, images: Images | None = None) -> CosmosReason2Endpoint:
    return CosmosReason2Endpoint(
        EndpointConfig(attempts=3, backoff_s=0.0), images or Images(), load_catalog(), transport, sleep=lambda _: None
    )


def test_pass_verdict_becomes_8b_production_evidence_for_the_same_capture() -> None:
    transport = Transport(_completion('{"verdict": "pass", "reason": "cover sits flat"}'))
    request = _request()
    result = _endpoint(transport).check(request)
    assert result.passed is True
    assert result.model_id == "Cosmos-Reason2-8B"
    assert result.evidence_source is CosmosEvidenceSource.COSMOS_REASON_2
    assert (result.request_id, result.image_sha256, result.observed_at) == (
        request.request_id,
        request.image_sha256,
        request.observed_at,
    )
    url, body = transport.calls[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert body["model"] == "Cosmos-Reason2-8B"
    image_url = body["messages"][1]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    prompt = body["messages"][1]["content"][1]["text"]
    assert "product 2" in prompt and "M2_04" in prompt and "<think>" in body["messages"][0]["content"]


def test_fail_verdict_is_a_rebuild_signal_not_an_error() -> None:
    result = _endpoint(Transport(_completion('{"verdict": "fail", "reason": "top part tilted"}'))).check(_request())
    assert result.passed is False and result.explanation == "top part tilted"


def test_answer_after_unparsed_think_block_is_accepted() -> None:
    content = '<think>gap visible</think>\n{"verdict": "fail", "reason": "raised"}'
    assert parse_completion(_completion(content, reasoning="")).passed is False


@pytest.mark.parametrize(
    "raw",
    [
        _completion('{"verdict": "pass", "reason": "ok"}', model="Cosmos-Reason2-2B"),
        _completion('{"verdict": "pass", "reason": "ok"}', finish="length"),
        _completion('{"verdict": "maybe", "reason": "unclear"}'),
        _completion('{"verdict": "pass"}'),
        _completion('{"verdict": "pass", "reason": "ok", "confidence": 1}'),
        _completion('{"verdict": "pass", "reason": "   "}'),
        _completion("pass"),
        _completion('```json\n{"verdict": "pass", "reason": "ok"}\n```'),
        b"not json",
        json.dumps({"model": MODEL_ID, "choices": []}).encode(),
    ],
)
def test_unverifiable_answers_are_unavailable_never_guessed(raw: bytes) -> None:
    with pytest.raises(CosmosReasonUnavailable):
        parse_completion(raw)


def test_image_digest_mismatch_is_unavailable_before_any_request() -> None:
    transport = Transport()
    with pytest.raises(CosmosReasonUnavailable, match="SHA-256"):
        _endpoint(transport, Images(JPEG + b"tampered")).check(_request())
    assert transport.calls == []


def test_missing_capture_and_unknown_format_are_unavailable() -> None:
    with pytest.raises(CosmosReasonUnavailable):
        _endpoint(Transport()).check(_request(image_capture_id="capture-2"))
    gif = b"GIF89a"
    with pytest.raises(CosmosReasonUnavailable, match="JPEG or PNG"):
        _endpoint(Transport(), Images(gif)).check(_request(image_sha256=hashlib.sha256(gif).hexdigest()))


def test_transport_failures_retry_three_times_then_hold() -> None:
    transport = Transport(OSError("refused"), TimeoutError(), OSError("HTTP 503"))
    with pytest.raises(CosmosReasonUnavailable, match="unreachable"):
        _endpoint(transport).check(_request())
    assert len(transport.calls) == 3


def test_transient_failure_then_success_returns_the_verdict() -> None:
    transport = Transport(OSError("refused"), _completion('{"verdict": "pass", "reason": "seated"}'))
    assert _endpoint(transport).check(_request()).passed is True


@pytest.mark.parametrize(
    "changes", [{"base_url": "ftp://host"}, {"attempts": 4}, {"max_tokens": 0}, {"temperature": -1.0}]
)
def test_config_rejects_unsafe_values(changes: dict) -> None:
    with pytest.raises(ValueError):
        EndpointConfig(**changes)


@pytest.mark.xfail(
    strict=True,
    reason="gate still pins Cosmos-Reason2-2B; coord/CODEX_BRIEF.md T35 asks Codex to pin 8B (drop this mark then)",
)
def test_gate_releases_a_recipe_complete_build_on_8b_pass_evidence() -> None:
    from agent.cosmos_reason import BuildCheckOutcome, FinalBuildGate

    transport = Transport(_completion('{"verdict": "pass", "reason": "all four parts seated"}'))
    decision = FinalBuildGate(_endpoint(transport), load_catalog()).evaluate(_request())
    assert decision.outcome is BuildCheckOutcome.DONE
