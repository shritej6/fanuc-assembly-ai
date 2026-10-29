"""Operator-facing service that routes text only into approved typed ports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from agent.commands import OperatorCommandError, OperatorCommandKind, parse_operator_command
from agent.cosmos_reason import BuildCheckOutcome, FinalBuildCheckError, FinalBuildGate, FinalGateDecision
from agent.task_port import LiveCellReport, OperatorTaskPort, OperatorTaskPortError, TaskReceipt

LOGGER = logging.getLogger(__name__)


class OperatorResponseKind(StrEnum):
    ACCEPTED = "accepted"
    REPORT = "report"
    FINAL_CHECK = "final_check"
    REJECTED = "rejected"
    HELD = "held"


@dataclass(frozen=True, slots=True)
class OperatorResponse:
    kind: OperatorResponseKind
    message: str
    receipt: TaskReceipt | None = None
    report: LiveCellReport | None = None
    final_check: FinalGateDecision | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "receipt": _receipt_dict(self.receipt),
            "report": None if self.report is None else self.report.as_dict(),
            "final_check": _final_check_dict(self.final_check),
        }


class OperatorAgent:
    """Compile operator text; all physical authority remains in the injected T09 port."""

    def __init__(self, tasks: OperatorTaskPort, final_gate: FinalBuildGate) -> None:
        if not isinstance(final_gate, FinalBuildGate):
            raise TypeError("final_gate must be a FinalBuildGate")
        required = ("submit", "report", "record_final_check")
        if any(not callable(getattr(tasks, name, None)) for name in required):
            raise TypeError("tasks must implement the OperatorTaskPort methods")
        self._tasks = tasks
        self._final_gate = final_gate

    def handle(self, text: object) -> OperatorResponse:
        try:
            command = parse_operator_command(text)
        except OperatorCommandError as error:
            return OperatorResponse(OperatorResponseKind.REJECTED, str(error))
        if command.kind is OperatorCommandKind.STATUS:
            return self._status()
        if command.task is None:
            return OperatorResponse(OperatorResponseKind.REJECTED, "assemble command has no typed task")
        try:
            receipt = self._tasks.submit(command.task)
        except OperatorTaskPortError as error:
            return OperatorResponse(OperatorResponseKind.HELD, str(error))
        except Exception:
            LOGGER.exception("operator task submission failed")
            return OperatorResponse(OperatorResponseKind.HELD, "operator task service failed; cell held")
        products = ", ".join(str(product) for product in receipt.selected_products)
        return OperatorResponse(OperatorResponseKind.ACCEPTED, f"accepted products {products}", receipt=receipt)

    def evaluate_final_build(self, request: object) -> OperatorResponse:
        try:
            decision = self._final_gate.evaluate(request)  # type: ignore[arg-type]
        except FinalBuildCheckError as error:
            return OperatorResponse(OperatorResponseKind.REJECTED, str(error))
        try:
            report = self._tasks.record_final_check(decision)
        except OperatorTaskPortError as error:
            return OperatorResponse(OperatorResponseKind.HELD, str(error), final_check=decision)
        except Exception:
            LOGGER.exception("final-build decision recording failed")
            message = "final-build decision could not be recorded; cell held"
            return OperatorResponse(OperatorResponseKind.HELD, message, final_check=decision)
        response_kind = (
            OperatorResponseKind.HELD
            if decision.outcome is BuildCheckOutcome.HOLD_ANDON
            else OperatorResponseKind.FINAL_CHECK
        )
        return OperatorResponse(
            response_kind,
            decision.detail,
            report=report,
            final_check=decision,
        )

    def _status(self) -> OperatorResponse:
        try:
            report = self._tasks.report()
        except OperatorTaskPortError as error:
            return OperatorResponse(OperatorResponseKind.HELD, str(error))
        except Exception:
            LOGGER.exception("operator status failed")
            return OperatorResponse(OperatorResponseKind.HELD, "operator status service failed; cell held")
        return OperatorResponse(OperatorResponseKind.REPORT, _report_message(report), report=report)


def _report_message(report: LiveCellReport) -> str:
    state = report.packml["state"]
    if report.task_id is None:
        return f"cell is {state}; no task is active"
    quality = report.quality_release.value.replace("_", " ")
    return f"cell is {state}; quality release is {quality}"


def _receipt_dict(receipt: TaskReceipt | None) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "task_id": receipt.task_id,
        "source": receipt.source.value,
        "selected_products": list(receipt.selected_products),
        "accepted_at": receipt.accepted_at,
    }


def _final_check_dict(decision: FinalGateDecision | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "product": decision.product,
        "task_id": decision.task_id,
        "request_id": decision.request_id,
        "observed_at": decision.observed_at.isoformat().replace("+00:00", "Z"),
        "outcome": decision.outcome.value,
        "recipe_passed": decision.recipe_passed,
        "cosmos_passed": decision.cosmos_passed,
        "evidence_source": None if decision.evidence_source is None else decision.evidence_source.value,
        "model_id": decision.model_id,
        "detail": decision.detail,
    }
