"""Tests for postprocess.
Authored by Shritej Gunda.
"""

import numpy as np

from sdg.postprocess import (
    apply_post,
    box_labels,
    depth_mm_u16,
    instance_mask,
    mask_box_labels,
    resize_nearest,
    uid_for_prim,
)

PATHS = {"/World/SdgScene/o00": "o00", "/World/SdgScene/o10": "o10"}
OBJECTS = {
    "o00": {"object_id": "F01", "kind": "base", "state": "on_table", "variant": "ok", "build_state": None},
    "o10": {"object_id": None, "foreign_type": "screw_m3", "kind": "foreign", "state": "on_table", "variant": "ok",
            "build_state": None},
}
BOX = np.dtype([("x_min", "i4"), ("y_min", "i4"), ("x_max", "i4"), ("y_max", "i4"), ("occlusionRatio", "f4")])


def test_child_prims_map_to_their_object_but_prefix_lookalikes_do_not():
    assert uid_for_prim("/World/SdgScene/o00/geom", PATHS) == "o00"
    assert uid_for_prim("/World/SdgScene/o001", PATHS) is None
    assert uid_for_prim("/World/ER4iA/J6_link", PATHS) is None


def test_instance_mask_keeps_only_scene_objects():
    ids = np.array([[1, 2], [3, 0]], dtype=np.uint32)
    mask = instance_mask(ids, {1: "/World/SdgScene/o00/geom", 2: "/World/ER4iA", 3: "/World/SdgScene/o10"}, PATHS)
    assert mask.tolist() == [[1, 0], [11, 0]]


def test_box_labels_join_the_spec_and_drop_slivers_and_non_scene_prims():
    boxes = np.array([(10, 10, 50, 40, 0.25), (0, 0, 1, 30, 0.0), (5, 5, 9, 9, 0.0)], dtype=BOX)
    labels = box_labels(boxes, ["/World/SdgScene/o00/geom", "/World/SdgScene/o10", "/World/Table"], PATHS, OBJECTS)
    assert labels == [{"uid": "o00", "object_id": "F01", "kind": "base", "state": "on_table", "variant": "ok",
                       "build_state": None, "bbox_xyxy": [10, 10, 50, 40], "occlusion": 0.25}]


def test_depth_is_millimetres_with_zero_for_no_return():
    d = depth_mm_u16(np.array([[0.5, np.inf], [np.nan, 0.3334]]))
    assert d.dtype == np.uint16 and d.tolist() == [[500, 0], [0, 333]]


def test_nearest_resize_never_invents_values():
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    small = resize_nearest(img, (2, 2))
    assert small.shape == (2, 2) and set(small.ravel()) <= set(img.ravel())


def test_post_is_deterministic_per_seed_and_off_means_identity():
    rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
    off = {"motion_blur": False, "gaussian_noise_sigma": 0.0}
    on = {"motion_blur": True, "gaussian_noise_sigma": 0.02}
    assert np.array_equal(apply_post(rgb, off, 1), rgb)
    assert np.array_equal(apply_post(rgb, on, 7), apply_post(rgb, on, 7))
    assert not np.array_equal(apply_post(rgb, on, 7), apply_post(rgb, on, 8))


def test_mask_boxes_are_tight_and_drop_slivers():
    mask = np.zeros((20, 30), dtype=np.uint16)
    mask[2:8, 5:15] = 1      # o00: 60 px
    mask[10, 0:5] = 11       # o10: 5 px sliver
    labels = mask_box_labels(mask, OBJECTS)
    assert labels == [{"uid": "o00", "object_id": "F01", "kind": "base", "state": "on_table", "variant": "ok",
                       "build_state": None, "bbox_xyxy": [5, 2, 15, 8], "visible_px": 60}]
