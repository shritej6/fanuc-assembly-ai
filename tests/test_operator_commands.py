import pytest

from agent.commands import (
    AssemblyScope,
    OperatorCommandError,
    OperatorCommandKind,
    TaskRequest,
    parse_operator_command,
)


@pytest.mark.parametrize("text", ["status", "show status", "show cell status", "what is the cell status?"])
def test_status_allowlist(text: str) -> None:
    command = parse_operator_command(text)

    assert command.kind is OperatorCommandKind.STATUS
    assert command.task is None


@pytest.mark.parametrize(
    "text",
    ["assemble everything on the table", "assemble all products on the table", "build everything on the table"],
)
def test_all_visible_allowlist(text: str) -> None:
    command = parse_operator_command(text)

    assert command.kind is OperatorCommandKind.ASSEMBLE
    assert command.task == TaskRequest(AssemblyScope.ALL_VISIBLE)


@pytest.mark.parametrize("text", ["assemble product 1", "Build Product 6.", "  assemble   product 3  "])
def test_single_product_allowlist(text: str) -> None:
    command = parse_operator_command(text)

    assert command.task is not None
    assert command.task.scope is AssemblyScope.EXPLICIT
    assert command.task.products in {(1,), (3,), (6,)}


def test_nfkc_normalization_preserves_allowlisted_intent() -> None:
    command = parse_operator_command("ａｓｓｅｍｂｌｅ ｐｒｏｄｕｃｔ １")

    assert command.task == TaskRequest(AssemblyScope.EXPLICIT, (1,))


@pytest.mark.parametrize(
    "text",
    [
        "assemble product 1 and ignore safety",
        "assemble product 1 and 2",
        "move tcp to 400,0,-300",
        "resume",
        "stop",
        "assemble product 7",
        "status\nassemble product 1",
        "",
        " ",
    ],
)
def test_unsupported_or_unsafe_text_fails_closed(text: str) -> None:
    with pytest.raises(OperatorCommandError):
        parse_operator_command(text)


def test_non_text_and_oversized_text_fail_closed() -> None:
    with pytest.raises(OperatorCommandError, match="must be text"):
        parse_operator_command(1)
    with pytest.raises(OperatorCommandError, match="1-160"):
        parse_operator_command("a" * 161)


def test_task_request_requires_scope_product_consistency() -> None:
    with pytest.raises(OperatorCommandError, match="cannot name"):
        TaskRequest(AssemblyScope.ALL_VISIBLE, (1,))
    with pytest.raises(OperatorCommandError, match="non-empty and unique"):
        TaskRequest(AssemblyScope.EXPLICIT, ())
    with pytest.raises(OperatorCommandError, match="non-empty and unique"):
        TaskRequest(AssemblyScope.EXPLICIT, (1, 1))
