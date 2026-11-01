"""Measured outcomes from scenario fleet runs.
Authored by Shritej Gunda."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from assembly.controller_ports import ControllerEvent
from eval.scenario_data import FaultKind


class ScenarioRunStatus(StrEnum):
    """Final status of one scenario run."""
    COMPLETE = "complete"
    HELD = "held"
    SUSPENDED = "suspended"
    ABORTED = "aborted"
    CALLBACK_LIMIT = "callback_limit"
    ERROR = "error"


class ScenarioPhysicalEventKind(StrEnum):
    """Kinds of physical event recorded during a run."""
    PRE_PICK = "pre_pick"
    REJECT_TRANSFER = "reject_transfer"


@dataclass(frozen=True, slots=True)
class ScenarioPhysicalEvent:
    """One physical event recorded during a run."""
    kind: ScenarioPhysicalEventKind
    instance_id: str


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Measured result of one scenario run."""
    scenario_id: str
    instruction: str
    products: tuple[int, ...]
    fault: FaultKind
    status: ScenarioRunStatus
    completed_products: tuple[int, ...]
    callbacks: int
    action_rows: int
    assists: int
    retries: int
    escapes: int | None
    unresolved_hazards: int
    physical_events: tuple[ScenarioPhysicalEvent, ...]
    events: tuple[ControllerEvent, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FleetReport:
    """Aggregate report over a list of scenario results."""
    results: tuple[ScenarioResult, ...]

    @property
    def scenario_count(self) -> int:
        """Return the number of scenarios run."""
        return len(self.results)

    @property
    def requested_products(self) -> int:
        """Return the number of products requested."""
        return sum(len(result.products) for result in self.results)

    @property
    def completed_products(self) -> int:
        """Return the number of products completed."""
        return sum(len(result.completed_products) for result in self.results)

    @property
    def action_rows(self) -> int:
        """Return the total number of action rows sent."""
        return sum(result.action_rows for result in self.results)

    @property
    def assists(self) -> int:
        """Return the total number of operator assists."""
        return sum(result.assists for result in self.results)

    @property
    def retries(self) -> int:
        """Return the total number of retries."""
        return sum(result.retries for result in self.results)

    @property
    def escapes(self) -> int | None:
        """Return the total escapes, or None if any run did not measure them."""
        values = [result.escapes for result in self.results]
        return None if any(value is None for value in values) else sum(value for value in values if value is not None)

    @property
    def known_escapes(self) -> int:
        """Return the total escapes, counting unmeasured runs as zero."""
        return sum(result.escapes or 0 for result in self.results)

    @property
    def unresolved_hazards(self) -> int:
        """Return the number of hazards left unresolved."""
        return sum(result.unresolved_hazards for result in self.results)

    @property
    def acceptance_ready(self) -> bool:
        """Return True when every run completed with zero escapes."""
        return bool(self.results) and all(
            result.status is ScenarioRunStatus.COMPLETE and result.escapes == 0 for result in self.results
        )

    @property
    def final_yield(self) -> float:
        """Return completed products divided by requested products."""
        return self.completed_products / self.requested_products if self.requested_products else 0.0

    @property
    def builds_per_assist(self) -> float | None:
        """Return completed builds per operator assist."""
        return None if self.assists == 0 else self.completed_products / self.assists

    def as_dict(self) -> dict[str, object]:
        """Return only the measured counters and ratios as a dict."""
        return {
            "scenario_count": self.scenario_count,
            "requested_products": self.requested_products,
            "completed_products": self.completed_products,
            "final_yield": self.final_yield,
            "action_rows": self.action_rows,
            "assists": self.assists,
            "builds_per_assist": self.builds_per_assist,
            "retries": self.retries,
            "escapes": self.escapes,
            "known_escapes": self.known_escapes,
            "unresolved_hazards": self.unresolved_hazards,
            "acceptance_ready": self.acceptance_ready,
            "status_counts": dict(Counter(result.status.value for result in self.results)),
            "fault_counts": dict(Counter(result.fault.value for result in self.results)),
        }
