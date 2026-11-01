"""Render the assembly instruction sheets for the 6 products from the product CAD.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ASSEMBLY_INSTRUCTIONS"
sys.path.insert(0, str(ROOT / "cad" / "products"))
MODULES = ("model_01_rotary", "model_02_indicator", "model_03_pushbutton",
           "model_04_sensor_connector", "model_05_io_connector", "model_06_power_connector")
NEW, DONE, BASE = "#e8742c", "#5b6270", "#9aa0aa"
EXPLODE_GAP = 22.0


def _solid(part) -> cq.Shape:
    return part.solid.val() if isinstance(part.solid, cq.Workplane) else part.solid


def _mesh(shape: cq.Shape) -> tuple[np.ndarray, np.ndarray]:
    v, t = shape.tessellate(0.08, 0.4)
    return np.array([p.toTuple() for p in v]), np.array(t)


def _view(ax, items, elev=24, azim=-58, lim=None) -> None:
    """One depth-sorted collection (separate collections are sorted only as whole objects)."""
    tris, colors = [], []
    for shape, c in items:
        v, t = _mesh(shape)
        tris.append(v[t])
        colors += [c] * len(t)
    ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=colors, edgecolors=colors, linewidths=0,
                                         shade=True, lightsource=LightSource(azdeg=90 - azim + 35, altdeg=elev + 40)))
    allv = np.vstack([t.reshape(-1, 3) for t in tris])
    lo, hi = lim if lim else (allv.min(axis=0) - 2, allv.max(axis=0) + 2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.array(hi) - np.array(lo)))
    ax.set_proj_type("ortho")
    ax.view_init(elev, azim)
    ax.set_axis_off()


def _limits(shapes) -> tuple[np.ndarray, np.ndarray]:
    bb = [s.BoundingBox() for s in shapes]
    lo = np.array([min(b.xmin for b in bb), min(b.ymin for b in bb), min(b.zmin for b in bb)]) - 2
    hi = np.array([max(b.xmax for b in bb), max(b.ymax for b in bb), max(b.zmax for b in bb)]) + 2
    return lo, hi


def _label(name: str) -> str:
    code, *words = name.split("_")[1:] if name.startswith("M") else name.split("_")
    return f"{name.split('_')[0]}_{code}  {' '.join(words)}" if name.startswith("M") else name.replace("_", " ")


def steps(parts, taxonomy: dict) -> list[dict]:
    """Return the part, gripper roll and look-alikes printed for each build step."""
    return [{"step": f"0{i}", "object_id": p.name[:5], "part": _label(p.name),
             "gripper": "turn gripper 90°" if p.grips and p.grips[0].axis == "Y" else "normal",
             "lookalikes": list(taxonomy[p.name[:5]]["lookalikes"])}
            for i, p in enumerate(parts, start=1)]


def product_page(pdf, mod, taxonomy: dict, out: Path = OUT) -> Path:
    fixture, parts = mod.build()
    base_id = fixture.name.split("_")[0]
    shapes = [_solid(fixture)] + [_solid(p) for p in parts]
    lim = _limits(shapes)
    title = mod.MODEL_DIR.replace("MODEL_0", "Product ").replace("_", " ", 2).replace("_", " ").title()
    fig = plt.figure(figsize=(16, 9.5))
    fig.suptitle(title, fontsize=19, weight="bold", x=0.02, ha="left")
    base_type = taxonomy[base_id].get("base_type", "?")
    fig.text(0.02, 0.905, f"Base: {fixture.name}  (base type {base_type}; any base of type {base_type} fits)   ·   "
             "Build 01 → 04.  Dismantle 04 → 01.   All real parts are black PLA (colours here are for clarity).",
             fontsize=11.5)
    for k in range(5):
        ax = fig.add_axes([0.01 + k * 0.196, 0.45, 0.196, 0.42], projection="3d")
        items = [(_solid(fixture), BASE if k else NEW)] + [(_solid(p), NEW if i == k - 1 else DONE)
                                                          for i, p in enumerate(parts[:k])]
        _view(ax, items, lim=lim)
        head = "0  Base on the table" if k == 0 else f"{k}  Add {_label(parts[k - 1].name)}"
        ax.set_title(head, fontsize=11, weight="bold")
    _exploded(fig, fixture, parts)
    _part_table(fig, parts, taxonomy)
    path = out / f"Product_{mod.MODEL_DIR[7]}.png"
    pdf.savefig(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _exploded(fig, fixture, parts) -> None:
    ax = fig.add_axes([0.0, 0.0, 0.42, 0.45], projection="3d")
    lifted = [_solid(p).translate(cq.Vector(0, 0, EXPLODE_GAP * (i + 1))) for i, p in enumerate(parts)]
    _view(ax, [(_solid(fixture), BASE)] + [(s, NEW if i % 2 == 0 else DONE) for i, s in enumerate(lifted)],
          elev=12, azim=-60)
    ax.set_title("Exploded: parts go straight down onto the base, in order", fontsize=11, weight="bold")
    ax.figure.canvas.draw()
    from mpl_toolkits.mplot3d import proj3d
    for i, s in enumerate(lifted):
        bb = s.BoundingBox()
        x, y, _ = proj3d.proj_transform(bb.xmax, bb.ymin, (bb.zmin + bb.zmax) / 2, ax.get_proj())
        ax.annotate(f"0{i + 1}", (x, y), xytext=(28, 0), textcoords="offset points", fontsize=13, weight="bold",
                    color=NEW if i % 2 == 0 else DONE, va="center")


def _part_table(fig, parts, taxonomy: dict) -> None:
    rows = [[s["step"], s["part"], s["gripper"], ", ".join(s["lookalikes"]) or "none"] for s in steps(parts, taxonomy)]
    ax = fig.add_axes([0.44, 0.06, 0.54, 0.33])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Step", "Part", "Gripper", "Look-alikes (check with wrist camera)"],
                     loc="upper left", cellLoc="left", colWidths=[0.08, 0.36, 0.2, 0.36])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    ax.text(0, -0.05, "Dismantle: remove 04, then 03, then 02, then 01 (straight up). The base never moves.",
            fontsize=11, transform=ax.transAxes)


def load_taxonomy() -> dict:
    return {o["object_id"]: o for o in json.loads((ROOT / "contracts" / "taxonomy.json").read_text())["objects"]}


def main(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("fanuc_cad").setLevel(logging.ERROR)
    taxonomy = load_taxonomy()
    with PdfPages(out / "Assembly_Instructions_All_Products.pdf") as pdf:
        for name in MODULES:
            print(product_page(pdf, importlib.import_module(name), taxonomy, out).name)


if __name__ == "__main__":
    main()
