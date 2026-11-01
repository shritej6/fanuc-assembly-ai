"""One-page ReportLab layout for validated birth-certificate records.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

_NAVY = HexColor("#17324D")
_BLUE = HexColor("#247BA0")
_PALE = HexColor("#EAF2F8")
_INK = HexColor("#1F2933")
_MUTED = HexColor("#52606D")
_GREEN = HexColor("#1B7F5A")
_AMBER = HexColor("#B7791F")
_RED = HexColor("#B42318")


def render_birth_certificate_pdf(record: dict[str, object], canonical_json_sha256: str) -> bytes:
    """Render one letter-sized page from an already validated record."""
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=letter, pageCompression=1)
    canvas.setTitle(f"Product {record['product']} birth certificate")
    canvas.setAuthor("Team V FANUC Assembly AI")
    canvas.setSubject("Product traceability record")
    _draw_header(canvas, record)
    _draw_identity(canvas, record)
    _draw_metrics(canvas, record)
    _draw_steps(canvas, record)
    _draw_final_checks(canvas, record)
    _draw_footer(canvas, canonical_json_sha256)
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def _draw_header(canvas: Canvas, record: dict[str, object]) -> None:
    canvas.setFillColor(_NAVY)
    canvas.rect(0, 708, letter[0], 84, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(42, 751, "PRODUCT BIRTH CERTIFICATE")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(42, 730, "Self-inspecting FANUC assembly cell - Team V")
    result = str(record["result"])
    canvas.setFillColor(_status_color(result))
    canvas.roundRect(421, 735, 149, 24, 4, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawCentredString(495.5, 743, result.replace("_", " ").upper())


def _draw_identity(canvas: Canvas, record: dict[str, object]) -> None:
    canvas.setFillColor(_INK)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(42, 681, f"Product {record['product']}")
    fields = (
        ("Run", str(record["run_id"])),
        ("Rollout", str(record["rollout"])),
        ("Base instance", str(record.get("base_instance") or "not assigned")),
    )
    x_positions = (42, 244, 376)
    for x, (label, value) in zip(x_positions, fields, strict=True):
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(x, 658, label.upper())
        canvas.setFillColor(_INK)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(x, 644, _fit(value, 180 if x == 42 else 190, "Helvetica", 9))


def _draw_metrics(canvas: Canvas, record: dict[str, object]) -> None:
    cycle = record.get("cycle_time_s")
    values = (
        ("REBUILDS", str(record.get("rebuilds", 0))),
        ("ASSISTS", str(record.get("assists", 0))),
        ("CYCLE TIME", "not measured" if cycle is None else f"{float(cycle):.2f} s"),
    )
    for index, (label, value) in enumerate(values):
        x = 42 + index * 176
        canvas.setFillColor(_PALE)
        canvas.roundRect(x, 588, 160, 42, 4, fill=1, stroke=0)
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(x + 10, 612, label)
        canvas.setFillColor(_INK)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(x + 10, 596, value)


def _draw_steps(canvas: Canvas, record: dict[str, object]) -> None:
    canvas.setFillColor(_INK)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(42, 561, "Assembly trace")
    columns = (42, 72, 126, 228, 302, 370, 420, 480, 570)
    labels = ("#", "Part", "Instance", "Verdict", "Checks", "Try", "Spare", "Evidence")
    canvas.setFillColor(_NAVY)
    canvas.rect(42, 531, 528, 22, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 7)
    for x, label in zip(columns[:-1], labels, strict=True):
        canvas.drawString(x + 4, 539, label)
    steps = record.get("steps", [])
    if not steps:
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.drawString(46, 507, "No assembly step was completed before the terminal outcome.")
        return
    for index, step in enumerate(steps):
        _draw_step_row(canvas, step, index, columns)


def _draw_step_row(canvas: Canvas, step: dict[str, object], index: int, columns: tuple[int, ...]) -> None:
    top = 531 - index * 43
    canvas.setFillColor(white if index % 2 == 0 else _PALE)
    canvas.rect(42, top - 43, 528, 43, fill=1, stroke=0)
    inspection = step.get("inspection", {})
    evidence = inspection.get("evidence", {}) if isinstance(inspection, dict) else {}
    values = (
        str(step.get("step", "")),
        str(step.get("object_id", "")),
        str(step.get("instance_id", "")),
        str(inspection.get("verdict", "")) if isinstance(inspection, dict) else "",
        str(step.get("seated_check", "")),
        str(step.get("attempts", "")),
        "yes" if step.get("spare_used") else "no",
        _score_summary(evidence),
    )
    widths = tuple(columns[i + 1] - columns[i] - 8 for i in range(len(values)))
    canvas.setFillColor(_INK)
    canvas.setFont("Helvetica", 7)
    for x, width, value in zip(columns[:-1], widths, values, strict=True):
        canvas.drawString(x + 4, top - 17, _fit(value, width, "Helvetica", 7))
    image_ref = evidence.get("image_ref") if isinstance(evidence, dict) else None
    if image_ref:
        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica", 6)
        canvas.drawString(484, top - 31, _fit(_basename(str(image_ref)), 82, "Helvetica", 6))


def _draw_final_checks(canvas: Canvas, record: dict[str, object]) -> None:
    canvas.setFillColor(_INK)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(42, 326, "Final poka-yoke checks")
    checks = record.get("final_checks", {})
    recipe = checks.get("recipe") if isinstance(checks, dict) else None
    cosmos = checks.get("cosmos_reason") if isinstance(checks, dict) else None
    _draw_check(canvas, 42, 286, "Recipe", recipe)
    _draw_check(canvas, 244, 286, "Cosmos Reason 2", cosmos)
    canvas.setFillColor(_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(42, 265, "A good product requires both independent final checks to pass.")


def _draw_check(canvas: Canvas, x: int, y: int, label: str, value: object) -> None:
    text = "NOT MEASURED" if value is None else ("PASS" if value is True else "FAIL")
    color = _AMBER if value is None else (_GREEN if value is True else _RED)
    canvas.setFillColor(color)
    canvas.roundRect(x, y, 184, 28, 4, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(x + 10, y + 10, f"{label}: {text}")


def _draw_footer(canvas: Canvas, canonical_json_sha256: str) -> None:
    canvas.setFillColor(_MUTED)
    canvas.setFont("Helvetica-Bold", 6)
    canvas.drawString(42, 82, "CANONICAL JSON SHA-256")
    canvas.setFont("Courier", 6)
    canvas.drawString(42, 71, canonical_json_sha256)
    canvas.setStrokeColor(_PALE)
    canvas.line(42, 62, 570, 62)
    canvas.setFillColor(_MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(42, 45, "Recorded evidence only. Missing measurements are shown as not measured.")
    canvas.drawRightString(570, 45, "Page 1 of 1")


def _score_summary(evidence: object) -> str:
    if not isinstance(evidence, dict):
        return "none"
    score_pairs = (
        ("C", evidence.get("classifier_score")),
        ("P", evidence.get("patchcore_score")),
        ("D", evidence.get("depth_residual_mm")),
    )
    values = [f"{label}:{float(value):.2f}" for label, value in score_pairs if isinstance(value, (int, float))]
    return " ".join(values) if values else "none"


def _fit(value: str, width: float, font: str, size: float) -> str:
    value = value.encode("ascii", errors="replace").decode("ascii")
    if stringWidth(value, font, size) <= width:
        return value
    candidate = value
    while candidate and stringWidth(candidate + "...", font, size) > width:
        candidate = candidate[:-1]
    return candidate + "..." if candidate else ""


def _basename(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).name


def _status_color(result: str):
    if result.startswith("good"):
        return _GREEN
    if result == "blocked":
        return _AMBER
    return _RED
