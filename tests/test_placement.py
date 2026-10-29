"""The real-cell placement: every rollout can be built, every object is reachable, and the file is what we submit."""

from __future__ import annotations

import math
import re

import yaml

from robot import placement as pl

SUBMISSION = pl.ROOT / "robot" / "cell" / "submission" / "submission.yaml"
OUT = pl.placement()
GEOMETRY, LAYOUT, _ = pl.load()


def test_the_committed_file_is_the_generator_output() -> None:
    assert yaml.safe_load(pl.OUT.read_text()) == OUT


def test_the_official_client_accepts_the_rollout_count() -> None:
    sub = yaml.safe_load(SUBMISSION.read_text())
    # run.py --submit: "different" objects or layout need one placement entry per rollout
    assert "different" in (sub["objects"], sub["layout"])
    assert len(OUT["rollouts"]) == sub["rollouts"] == len(pl.ROLLOUTS)


def test_every_product_has_its_base_and_four_parts_and_every_replaced_part_has_a_spare() -> None:
    for spec, rollout in zip(pl.ROLLOUTS, OUT["rollouts"], strict=True):
        labels = list(rollout["objects"])
        for p in spec["products"]:
            assert sum(label.startswith(f"F0{p} base") for label in labels) == 1
            for i in range(1, 5):
                oid = f"M{p}_0{i}"
                usable = [lb for lb in labels if lb.startswith(f"{oid} ")]
                faulty = [lb for lb in labels if re.match(rf"(DEFECT|FALLEN) {oid} ", lb)]
                spare = [lb for lb in labels if lb.startswith(f"spare {oid} ")]
                assert len(usable) == 1 or (faulty and spare), (spec["products"], oid)


def test_traps_belong_to_no_product_in_their_rollout() -> None:
    for spec in pl.ROLLOUTS:
        for kind, oid, _ in spec["faults"]:
            if kind == "TRAP":
                assert int(oid[1]) not in spec["products"]


def test_loose_objects_keep_their_gaps_and_spares_sit_in_store_slots() -> None:
    slots = {(s["x"], s["y"]) for s in LAYOUT["spare_store"]["slots"]}
    for rollout in OUT["rollouts"]:
        placed = []
        for label, (x, y) in rollout["objects"].items():
            if label.startswith("spare "):
                assert (x, y) in slots
                continue
            kind = "base" if " base (" in label else "part"
            oid = label.split()[0] if kind == "base" else re.search(r"M\d_0\d", label).group(0)
            r = GEOMETRY[oid]["upright"]["footprint_r"] * (pl.FALLEN_SCALE if label.startswith("FALLEN") else 1)
            assert pl._inside(x, y, r, LAYOUT), label
            for px, py, pr, pk in placed:
                assert math.hypot(x - px, y - py) >= r + pr + pl._gap(kind, pk) - 1, label   # 1 mm: rounding
            placed.append((x, y, r, kind))


def test_every_object_is_reachable_straight_down_at_every_gripper_roll() -> None:
    assert pl.unreachable(OUT) == []


def test_an_object_out_of_reach_is_reported() -> None:
    far = {"rollouts": [{"objects": {"M1_01 Housing": [900, 0]}}]}

    assert pl.unreachable(far) == ["rollout 1 M1_01 Housing"]
