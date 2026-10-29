"""T28: stable part ids across frames."""

from __future__ import annotations

import pytest

from perception.cell.tracker import GATE_MM, MISS_FRAMES, Detection, Tracker, TrackerError


def _det(label: str, x: float, y: float) -> Detection:
    return Detection(label=label, score=0.9, x_mm=x, y_mm=y, top_z_mm=-310.0, bbox_px=(0, 0, 10, 10))


def test_ids_stay_stable_under_jitter_and_order_changes():
    tracker = Tracker()
    first = {t.label: t.instance_id for t in tracker.update([_det("M1_01", 300, 0), _det("M2_01", 400, 100)])}
    for dx in (1.5, -2.0, 2.5):
        seen = tracker.update([_det("M2_01", 400 - dx, 100 + dx), _det("M1_01", 300 + dx, 0)])
        assert {t.label: t.instance_id for t in seen} == first


def test_label_is_the_majority_of_recent_frames_not_the_last_one():
    tracker = Tracker()
    for _ in range(5):
        tracker.update([_det("M5_04", 300, 0)])
    (track,) = tracker.update([_det("M4_04", 300, 0)])
    assert track.label == "M5_04" and track.votes == (5, 6)


def test_a_part_that_falls_keeps_its_id_and_relabels_after_enough_frames():
    tracker = Tracker()
    (track,) = tracker.update([_det("M3_03", 300, 0)])
    for _ in range(9):
        (later,) = tracker.update([_det("M3_03__fallen", 301, 0)])
    assert later.instance_id == track.instance_id and later.label == "M3_03__fallen"


def test_far_detection_is_a_new_part():
    tracker = Tracker()
    tracker.update([_det("M1_01", 300, 0)])
    (seen,) = tracker.update([_det("M1_01", 300 + GATE_MM + 1, 0)])
    assert seen.instance_id == "p2"


def test_missed_track_is_dropped_only_after_miss_frames():
    tracker = Tracker()
    tracker.update([_det("M1_01", 300, 0)])
    for _ in range(MISS_FRAMES - 1):
        tracker.update([])
    assert [t.instance_id for t in tracker.tracks()] == ["p1"]
    tracker.update([])
    assert tracker.tracks() == ()


def test_two_close_detections_never_share_a_track():
    tracker = Tracker()
    tracker.update([_det("M1_01", 300, 0)])
    seen = tracker.update([_det("M1_01", 301, 0), _det("M1_02", 305, 0)])
    assert len({t.instance_id for t in seen}) == 2


def test_moved_part_keeps_its_id_at_the_new_place():
    tracker = Tracker()
    (track,) = tracker.update([_det("M4_04", 300, 0)])
    tracker.moved(track.instance_id, 250, 350)
    (seen,) = tracker.update([_det("M4_04", 251, 349)])
    assert seen.instance_id == track.instance_id


def test_moved_rejects_unknown_track_and_non_finite_position():
    tracker = Tracker()
    with pytest.raises(TrackerError, match="no track"):
        tracker.moved("p9", 1, 2)
    (track,) = tracker.update([_det("M1_01", 300, 0)])
    with pytest.raises(TrackerError, match="non-finite"):
        tracker.moved(track.instance_id, float("nan"), 2)
