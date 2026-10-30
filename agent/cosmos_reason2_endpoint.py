"""Production Cosmos Reason 2 (8B) port for the final-build gate, over vLLM's OpenAI-compatible API.

The endpoint is the official ``nvidia/Cosmos-Reason2-8B`` weights at a pinned revision, served by vLLM with
``--served-model-name Cosmos-Reason2-8B`` (serving/cosmos_reason2/serve.sh). The server runs without a reasoning
parser: the model may or may not open a ``<think>`` block, and vLLM's qwen3 parser files a think-less answer as
reasoning with no content (seen on 2026-10-30), so the port splits ``<think>...</think>`` itself. No JSON-schema
constraint is sent either, because it would force JSON from the first token and remove the reasoning. The
port sends the captured image (looked up by capture ID, its SHA-256 re-checked) and a recipe-derived prompt, and turns
the model's final JSON answer into a ``CosmosReasonResult``. Anything it cannot verify, such as a digest mismatch,
transport failure, truncated or unparseable answer or a different served model, raises ``CosmosReasonUnavailable``,
which the gate turns into Hold/Andon. It never guesses a verdict.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent.cosmos_reason import (
    CosmosEvidenceSource,
    CosmosReasonResult,
    CosmosReasonUnavailable,
    FinalBuildRequest,
)
from assembly.catalog import Catalog

LOGGER = logging.getLogger(__name__)

MODEL_REPO = "nvidia/Cosmos-Reason2-8B"
MODEL_REVISION = "a9fae2cf89dc64db96b12860417f0eb403013bb9"
MODEL_ID = "Cosmos-Reason2-8B"
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "contracts" / "taxonomy.json"

# NVIDIA's reasoning add-on (cosmos-reason2 cosmos_reason2_utils/text.py), appended to the default system prompt.
SYSTEM_PROMPT = """You are a helpful assistant.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""

Transport = Callable[[str, bytes, float], bytes]


class CaptureImageSource(Protocol):
    """Trusted capture store: returns the encoded image (JPEG or PNG) for a capture ID."""

    def image_bytes(self, capture_id: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    timeout_s: float = 120.0
    # Measured on the Spark (2026-10-30, 39 set D answers): reasoning <= ~500 tokens, <= 39 s; one crop looped to
    # 4096 tokens in 299 s. 1024 tokens (~75 s at ~14 tok/s) bounds that case; a cut-off answer is unavailable (Hold).
    max_tokens: int = 1024
    temperature: float = 0.0
    attempts: int = 3
    backoff_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        if self.timeout_s <= 0 or self.max_tokens <= 0 or not 1 <= self.attempts <= 3 or self.backoff_s < 0:
            raise ValueError("timeout, max_tokens, attempts (1-3) and backoff must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be within 0-2")


@dataclass(frozen=True, slots=True)
class BuildVerdict:
    passed: bool
    reason: str
    reasoning: str


class CosmosReason2Endpoint:
    """``CosmosReasonPort`` backed by a live Cosmos Reason 2 8B vLLM server."""

    def __init__(
        self,
        config: EndpointConfig,
        images: CaptureImageSource,
        catalog: Catalog,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._images = images
        self._catalog = catalog
        self._transport = transport or _http_post
        self._sleep = sleep
        self._families = _load_families(TAXONOMY_PATH)
        self.last_verdict: BuildVerdict | None = None

    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        image = _verified_image(self._images, request)
        prompt = build_prompt(self._catalog, self._families, request)
        payload = build_payload(prompt, image, self._config)
        verdict = parse_completion(self._post(payload, request.request_id))
        self.last_verdict = verdict
        return CosmosReasonResult(
            request.run_id,
            request.rollout,
            request.task_id,
            request.request_id,
            request.product,
            request.image_capture_id,
            request.image_sha256,
            verdict.passed,
            verdict.reason,
            MODEL_ID,
            CosmosEvidenceSource.COSMOS_REASON_2,
            request.observed_at,
        )

    def _post(self, payload: bytes, request_id: str) -> bytes:
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(1, self._config.attempts + 1):
            try:
                return self._transport(url, payload, self._config.timeout_s)
            except (OSError, TimeoutError) as error:
                LOGGER.warning(
                    "Cosmos Reason 2 transport failed",
                    extra={"request_id": request_id, "attempt": attempt, "error": type(error).__name__},
                )
                if _timed_out(error):
                    raise CosmosReasonUnavailable("Cosmos Reason 2 answer timed out") from error
                if attempt == self._config.attempts:
                    raise CosmosReasonUnavailable("Cosmos Reason 2 endpoint unreachable") from error
                self._sleep(self._config.backoff_s * 2 ** (attempt - 1))
        raise CosmosReasonUnavailable("Cosmos Reason 2 endpoint unreachable")


def build_prompt(catalog: Catalog, families: Mapping[str, str], request: FinalBuildRequest) -> str:
    recipe = catalog.product(request.product)
    installed = request.installed_object_ids
    if not installed:
        raise CosmosReasonUnavailable("no installed parts to check")
    layers = "\n".join(
        f"{index}. {object_id} ({_readable(families, object_id)})" for index, object_id in enumerate(installed, 1)
    )
    top = installed[-1]
    base = _readable(families, _base_of(families, recipe.product))
    return (
        f"The image shows product {recipe.product}, the {recipe.name}, on the work table of a robotic assembly cell. "
        f"All parts are black 3D-printed plastic. It is built on a {base} fixture; these parts should be installed "
        f"on it, bottom to top:\n{layers}\n\n"
        f"Check the build, especially the top part, {top} ({_readable(families, top)}). The build passes only if the "
        "top part is the right part for this product, sits fully seated and flat on the part below it with no visible "
        "gap, and is not tilted. It fails if the top part sits high (raised by a few millimetres), is tilted, or is a "
        "different part.\n\n"
        'After </think>, answer with only this JSON: {"verdict": "pass" or "fail", "reason": "<one short sentence>"}'
    )


def build_payload(prompt: str, image: bytes, config: EndpointConfig) -> bytes:
    mime = _image_mime(image)
    data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    body = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "seed": 0,
    }
    return json.dumps(body).encode("utf-8")


def parse_completion(raw: bytes) -> BuildVerdict:
    try:
        body = json.loads(raw)
        choice = body["choices"][0]
        message = choice["message"]
        model = body["model"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise CosmosReasonUnavailable("Cosmos Reason 2 response is not a chat completion") from error
    if model != MODEL_ID:
        raise CosmosReasonUnavailable(f"endpoint served {model!r}, not {MODEL_ID}")
    if choice.get("finish_reason") != "stop":
        raise CosmosReasonUnavailable(f"Cosmos Reason 2 answer did not finish ({choice.get('finish_reason')!r})")
    reasoning, answer = _split_think(message.get("content"))
    return _parse_answer(answer, reasoning)


def _parse_answer(answer: str, reasoning: str) -> BuildVerdict:
    fenced = _FENCE.fullmatch(answer)
    try:
        verdict = json.loads(fenced.group(1) if fenced else answer)
    except ValueError as error:
        raise CosmosReasonUnavailable("Cosmos Reason 2 final answer is not JSON") from error
    if not isinstance(verdict, dict) or set(verdict) != {"verdict", "reason"}:
        raise CosmosReasonUnavailable("Cosmos Reason 2 answer must have exactly verdict and reason")
    label, reason = verdict["verdict"], verdict["reason"]
    if label not in ("pass", "fail") or not isinstance(reason, str) or not reason.strip():
        raise CosmosReasonUnavailable("Cosmos Reason 2 verdict must be pass or fail with a reason")
    if len(reason) > 2048 or any(not character.isprintable() for character in reason):
        raise CosmosReasonUnavailable("Cosmos Reason 2 reason must be short printable text")
    return BuildVerdict(label == "pass", reason.strip(), reasoning)


def _split_think(content: object) -> tuple[str, str]:
    if not isinstance(content, str) or not content.strip():
        raise CosmosReasonUnavailable("Cosmos Reason 2 returned no final answer")
    text = content.strip()
    if "</think>" not in text:
        return "", text
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "", 1).strip(), answer.strip()


def _verified_image(images: CaptureImageSource, request: FinalBuildRequest) -> bytes:
    try:
        image = images.image_bytes(request.image_capture_id)
    except (OSError, KeyError) as error:
        raise CosmosReasonUnavailable(f"capture {request.image_capture_id!r} is not available") from error
    if not isinstance(image, bytes) or hashlib.sha256(image).hexdigest() != request.image_sha256:
        raise CosmosReasonUnavailable("captured image does not match its recorded SHA-256")
    return image


def _image_mime(image: bytes) -> str:
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise CosmosReasonUnavailable("captured image must be JPEG or PNG")


def _timed_out(error: BaseException) -> bool:
    # A timed-out generation would only repeat (seed 0, temperature 0): hold at once instead of waiting 3x.
    return isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError)


def _http_post(url: str, payload: bytes, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code >= 500 or error.code == 429:
            raise OSError(f"HTTP {error.code}") from error
        raise CosmosReasonUnavailable(f"Cosmos Reason 2 rejected the request (HTTP {error.code})") from error


def _load_families(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["object_id"]: entry["family"] for entry in data["objects"]}


def _base_of(families: Mapping[str, str], product: int) -> str:
    return f"F{product:02d}" if f"F{product:02d}" in families else "base"


def _readable(families: Mapping[str, str], object_id: str) -> str:
    return families.get(object_id, object_id).replace("_", " ")


class DirectoryImageSource:
    """Capture store backed by ``<root>/<capture_id>.jpg`` or ``.png`` files (bench and health check)."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def image_bytes(self, capture_id: str) -> bytes:
        for suffix in (".jpg", ".png"):
            path = self._root / f"{capture_id}{suffix}"
            if path.is_file():
                return path.read_bytes()
        raise KeyError(capture_id)
