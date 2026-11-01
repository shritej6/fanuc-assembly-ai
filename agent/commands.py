"""Strict natural-language compiler for the operator's approved task vocabulary.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class OperatorCommandError(ValueError):
    """Operator text is malformed, ambiguous, or outside the allowlist."""


class OperatorCommandKind(StrEnum):
    ASSEMBLE = "assemble"
    STATUS = "status"


class AssemblyScope(StrEnum):
    ALL_VISIBLE = "all_visible"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class TaskRequest:
    scope: AssemblyScope
    products: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AssemblyScope):
            raise OperatorCommandError("task scope must be an AssemblyScope")
        if not isinstance(self.products, tuple):
            raise OperatorCommandError("task products must be a tuple")
        if self.scope is AssemblyScope.ALL_VISIBLE and self.products:
            raise OperatorCommandError("all-visible requests cannot name explicit products")
        if self.scope is AssemblyScope.EXPLICIT:
            _validate_products(self.products)


@dataclass(frozen=True, slots=True)
class OperatorCommand:
    kind: OperatorCommandKind
    task: TaskRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperatorCommandKind):
            raise OperatorCommandError("command kind must be an OperatorCommandKind")
        if (self.kind is OperatorCommandKind.ASSEMBLE) != (self.task is not None):
            raise OperatorCommandError("only assemble commands carry a task request")


_STATUS_COMMANDS = frozenset(
    {
        "status",
        "show status",
        "show cell status",
        "what is the cell status?",
    }
)
_ALL_VISIBLE_COMMANDS = frozenset(
    {
        "assemble everything on the table",
        "assemble all products on the table",
        "build everything on the table",
    }
)
_SINGLE_PRODUCT = re.compile(r"(?:assemble|build) product ([1-6])\.?", re.ASCII)
_MAX_COMMAND_CHARS = 160


def parse_operator_command(text: object) -> OperatorCommand:
    """Compile an exact approved phrase; never infer intent from unsupported text."""
    normalized = _normalize(text)
    if normalized in _STATUS_COMMANDS:
        return OperatorCommand(OperatorCommandKind.STATUS)
    if normalized in _ALL_VISIBLE_COMMANDS:
        task = TaskRequest(AssemblyScope.ALL_VISIBLE)
        return OperatorCommand(OperatorCommandKind.ASSEMBLE, task)
    match = _SINGLE_PRODUCT.fullmatch(normalized)
    if match is not None:
        task = TaskRequest(AssemblyScope.EXPLICIT, (int(match.group(1)),))
        return OperatorCommand(OperatorCommandKind.ASSEMBLE, task)
    raise OperatorCommandError(
        "unsupported or ambiguous command; use status, assemble product 1-6, or assemble everything on the table"
    )


def _normalize(text: object) -> str:
    if not isinstance(text, str):
        raise OperatorCommandError("operator command must be text")
    if not text or len(text) > _MAX_COMMAND_CHARS:
        raise OperatorCommandError(f"operator command must contain 1-{_MAX_COMMAND_CHARS} characters")
    normalized = unicodedata.normalize("NFKC", text)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise OperatorCommandError("operator command contains a control character")
    collapsed = " ".join(normalized.strip().lower().split())
    if not collapsed:
        raise OperatorCommandError("operator command cannot be blank")
    return collapsed


def _validate_products(products: tuple[int, ...]) -> None:
    if not products or len(set(products)) != len(products):
        raise OperatorCommandError("explicit product selection must be non-empty and unique")
    if any(isinstance(product, bool) or not isinstance(product, int) or not 1 <= product <= 6 for product in products):
        raise OperatorCommandError("product IDs must be integers from 1 through 6")
