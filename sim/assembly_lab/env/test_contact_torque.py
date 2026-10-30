"""Seating torque from non-robot contacts (contact_torque.py), on numpy arrays."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contact_torque as T


def _empty(n, m):
    z = np.zeros((n, m), int)
    return np.zeros((0, 1)), np.zeros((0, 3)), np.zeros((0, 3)), z, z, np.zeros((0, 3)), np.zeros((0, 3)), z, z


def test_part_held_in_the_air_reads_zero_torque():
    fn, p, nrm, c, s, fr, fp, fc, fs = _empty(2, 3)
    tz, tau, force = T.env_torque(fn, p, nrm, c, s, fr, fp, fc, fs, np.zeros((2, 3)), np.tile([0, 0, 1.0], (2, 1)))
    assert np.all(tz == 0.0) and np.all(tau == 0.0) and np.all(force == 0.0)


def test_friction_couple_about_the_insert_axis():
    # body 1 of 2, filter 0: two friction anchors 10 mm either side of the COM pushing opposite ways in y
    fn, p, nrm, c, s, _, _, fc, fs = _empty(2, 2)
    fr = np.array([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]])
    fp = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    fc, fs = fc.copy(), fs.copy()
    fc[1, 0], fs[1, 0] = 2, 0
    tz, _tau, force = T.env_torque(fn, p, nrm, c, s, fr, fp, fc, fs, np.zeros((2, 3)), np.tile([0, 0, 1.0], (2, 1)))
    assert np.isclose(tz[1], 0.04) and tz[0] == 0.0
    assert np.allclose(force[1], 0.0)


def test_resting_weight_gives_no_insert_axis_torque():
    # four normal contacts at the corners of a 20 mm square, 0.05 N each, straight up
    fn = np.full((4, 1), 0.05)
    p = np.array([[0.01, 0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [-0.01, -0.01, 0]])
    nrm = np.tile([0, 0, 1.0], (4, 1))
    c, s = np.array([[4]]), np.array([[0]])
    z = np.zeros((1, 1), int)
    tz, _, force = T.env_torque(fn, p, nrm, c, s, np.zeros((0, 3)), np.zeros((0, 3)), z, z,
                                np.array([[0, 0, 0.005]]), np.array([[0, 0, 1.0]]))
    assert np.isclose(tz[0], 0.0) and np.allclose(force[0], [0, 0, 0.2])


def test_contact_point_off_the_part_is_dropped():
    # one real corner contact plus a buffer artefact 1 m away carrying a lateral force: the artefact must not count
    fn = np.array([[0.05], [0.0]])
    p = np.array([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nrm = np.array([[0, 0, 1.0], [0, 0, 1.0]])
    c, s = np.array([[1]]), np.array([[0]])
    fr, fp = np.array([[0.0, 2.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    fc, fs = np.array([[1]]), np.array([[0]])
    tz, _, force = T.env_torque(fn, p, nrm, c, s, fr, fp, fc, fs, np.zeros((1, 3)), np.array([[0, 0, 1.0]]),
                                max_arm=0.1)
    assert tz[0] == 0.0 and np.allclose(force[0], [0, 0, 0.05])


def test_resting_untouched_part_reads_zero_guard_torque_for_100_steps():
    # object 0 rests on the table (filter 1) far from the robot, with the spurious friction couple measured in sim
    # (+-0.035 N anchors 10 mm apart); nobody touches it, so the guard reads 0 every step while the raw value is logged
    rng = np.random.default_rng(0)
    for _ in range(100):
        f = 0.035 + rng.normal(0, 0.01)
        fr = np.array([[0.0, f, 0.0], [0.0, -f, 0.0]])
        fp = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        fn, p, nrm, c, s, _, _, fc, fs = _empty(1, 2)
        fc, fs = fc.copy(), fs.copy()
        fc[0, 1] = 2
        raw, _, _ = T.env_torque(fn, p, nrm, c, s, fr, fp, fc, fs, np.zeros((1, 3)), np.array([[0, 0, 1.0]]))
        loaded = T.robot_loaded(np.array([0.0]), np.zeros((1, 1)))
        assert raw[0] > 0.0 and T.guard_torque(raw, loaded)[0] <= 0.005


def test_part_twisted_by_the_robot_on_a_peg_reads_the_peg_torque():
    # object 0 is gripped (robot touch 3 N) and twisted on its base, object 1: the base's peg pushes back with a
    # 0.06 N m couple on object 0 (anchors 10 mm apart, 3 N each way); the base touches the held part, so it is
    # loaded too and reads the equal and opposite couple; an untouched part 2 resting nearby reads 0
    fn, p, nrm, c, s, _, _, fc, fs = _empty(3, 4)
    fr = np.array([[0.0, 3.0, 0.0], [0.0, -3.0, 0.0], [0.0, -3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.5, 0.0],
                   [0.0, -0.5, 0.0]])
    fp = np.array([[0.01, 0, 0], [-0.01, 0, 0], [0.01, 0, 0], [-0.01, 0, 0], [0.31, 0, 0], [0.29, 0, 0]])
    fc, fs = fc.copy(), fs.copy()
    fc[0, 1], fs[0, 1] = 2, 0          # object 0 against object 1
    fc[1, 0], fs[1, 0] = 2, 2          # object 1 against object 0
    fc[2, 3], fs[2, 3] = 2, 4          # object 2 against the table
    com = np.array([[0.0, 0, 0], [0.0, 0, 0], [0.3, 0, 0]])
    raw, _, _ = T.env_torque(fn, p, nrm, c, s, fr, fp, fc, fs, com, np.tile([0, 0, 1.0], (3, 1)))
    pair = np.zeros((3, 3))
    pair[0, 1] = pair[1, 0] = 6.0
    loaded = T.robot_loaded(np.array([3.0, 0.0, 0.0]), pair)
    guard = T.guard_torque(raw, loaded)
    assert np.allclose(guard, [0.06, 0.06, 0.0]) and raw[2] > 0.0


def test_robot_loaded_per_env_matches_each_env_alone():
    rng = np.random.default_rng(3)
    touch = rng.choice([0.0, 0.5], size=(4, 5))
    pair = rng.choice([0.0, 0.2], size=(4, 5, 5))
    batched = T.robot_loaded(touch, pair)
    assert batched.shape == (4, 5)
    for e in range(4):
        assert np.array_equal(batched[e], T.robot_loaded(touch[e], pair[e]))


def test_env_major_buffer_gives_each_env_its_own_torque():
    # 2 envs x 2 bodies, flattened env-major as the contact view reports them; only env 1 body 0 has a couple
    fn, p, nrm, c, s, _, _, fc, fs = _empty(4, 1)
    fr = np.array([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]])
    fp = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]) + [5.0, 0.0, 0.0]      # env 1 sits 5 m along x
    fc, fs = fc.copy(), fs.copy()
    fc[2, 0], fs[2, 0] = 2, 0
    com = np.array([[0.0, 0, 0], [0.1, 0, 0], [5.0, 0, 0], [5.1, 0, 0]])
    tz, _, _ = T.env_torque(fn, p, nrm, c, s, fr, fp, fc, fs, com, np.tile([0, 0, 1.0], (4, 1)), max_arm=0.1)
    assert np.allclose(tz.reshape(2, 2), [[0.0, 0.0], [0.04, 0.0]])
