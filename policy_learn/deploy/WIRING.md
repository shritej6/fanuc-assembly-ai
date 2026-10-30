# Wiring GR00T into `robot/cell/policy.py` (for Codex, T05)

Agent 6 does not edit `robot/cell/*`. This note says how `policy.py` should call the GR00T adapter.

## What exists

| File | What it is |
|---|---|
| `policy_learn/deploy/groot_policy.py` | `GrootPolicy`: the official interface (`CAMERAS`, `IMAGE_SIZE`, `ACTION = "tcp"`, `DT = 0.2`, `PREFETCH = False`, `reset()`, `act(obs)`) plus `set_plan(skill, object_id, base_id, grip, case, target, ContactTarget)` for the planner |
| `policy_learn/deploy/guards.py` | `ChunkGuard`: every chunk is checked before it leaves `act()`. A violation returns one hold row (current pose, gripper unchanged) and a reason; a small overspeed is clamped, with a reason |
| `policy_learn/groot/groot_client.py` | Agent 5's ZMQ client to the GR00T policy server (eef_9d + gripper, dt 0.2 s) |
| `policy_learn/deploy/mock_check.py` | Runs `run.py --mock` with `GrootPolicy` swapped in, without editing `robot/cell` |

## policy.py (proposed)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for policy_learn/

from policy_learn.deploy.groot_policy import GrootPolicy
from policy_learn.deploy.guards import ContactTarget


class Policy(GrootPolicy):
    """checkpoint: "groot://127.0.0.1:5555" (the GR00T server, through an ssh tunnel); None reads $GROOT_POLICY."""

    def act(self, obs):
        step = self.planner.next(obs)  # T09 state machine: the current step and its contact target
        self.set_plan(step.skill, step.object_id, step.base_id, step.grip, step.case,
                      [step.part_x, step.part_y, step.part_yaw, step.roll, step.goal_x, step.goal_y, step.goal_yaw],
                      ContactTarget(step.x, step.y, step.z_contact, label=step.label)
                      if step.z_contact is not None else None)
        return super().act(obs)
```

Rules:

- **Keep `ACTION = "tcp"` and `DT = 0.2`.** GR00T is trained at 5 Hz. If the chunk is played at another dt, the speed
  the guard checks no longer matches the speed of the arm.
- **Keep `PREFETCH = False`.** The guard checks each chunk against the pose it starts from. With prefetch, that pose
  would be stale.
- **Call `set_plan` before every `act`.**
  - It builds the instruction with `sim/assembly_lab/mimic/instructions.instruction`, the function the demos were
    labelled with. An example: "pick M4_02 at (300, -50, 37) gripping X at roll 127, insert on F04 at (400, 50, 90)".
  - Do not hand-write instructions with `set_task`. Different words from the training text make GR00T unreliable.
  - `skill` is one of pick_insert, pull_out, place_free or defect_drop. `case` comes from
    `sim/assembly_lab/mimic/catalog.CASES`.
  - Pass the `ContactTarget` of the current insert or grasp:
    - `z_contact_mm` is the TCP z at the seated pose (CAD seated height).
    - The guard refuses anything lower than `z_contact_mm - 1` within `radius_mm` (25 mm) of `(x_mm, y_mm)`.
    - The last 5 mm above `z_contact_mm` runs at the slow limit.
  - With no target, only the box, tilt, speed, table slow zone and gripper checks apply.
- **Never bypass the guard.** Do not send `model.predict` rows yourself. `GrootPolicy.guard_log` lists every hold and
  clamp (rollout, act number, reasons). Log it with the run.
- **A hold is not an error.**
  - The arm stays put and `act` is called again with a fresh observation.
  - If the guard holds three times in a row, the state machine should treat it as "stuck": re-scan and re-plan, then
    Andon. (Not implemented in the adapter: the state machine owns that decision.)
- **`tcp_pose` missing.** The mock sends none before the first tcp command. `act` then returns one hold row at the
  FK-estimated pose and does not call the model. The real cell always sends `tcp_pose`.
- **Angles near r = ±180.** A decoded rotation flips between r = 179.8 and −179.9 although the fingers barely turn.
  `act` rewrites each row's w and r as the same orientation, numerically nearest the row before, starting from the
  reported pose (`nearest_turns`). Without this, the guard holds (it reads 179 → −179 as 358° of J6). Found in the
  Isaac smoke: the arm froze on its first move toward a grasp at r ≈ 180. A genuinely large turn is still refused.

## How to run

```bash
# plumbing check, no model
cd robot/cell && GROOT_POLICY=stub uv run python ../../policy_learn/deploy/mock_check.py
# with the real GR00T server (agent 5's serve_policy.sh on the training box, tunnelled to :5555)
ssh -N -L 5555:127.0.0.1:5555 fa-train &
cd robot/cell && GROOT_POLICY=groot://127.0.0.1:5555 \
  uv run --with pyzmq --with msgpack --with msgpack-numpy python ../../policy_learn/deploy/mock_check.py
```

`robot/cell/pyproject.toml` has no pyzmq or msgpack. Either T05 adds `pyzmq`, `msgpack` and `msgpack-numpy` there, or
the run uses `uv run --with ...` as above.

## Guard limits (`GuardLimits`)

- Workspace box x 150–650, y −450 to 450, z −334 to 300 mm; dt 0.1–1.0 s; |w, p, r| ≤ 360. These are the official
  limits.
- Fingers-down cone 6°. This covers the 3° insert tilt plus the ±2° twist.
- The speeds are **PRELIMINARY choices, not measured**:
  - 150 mm/s and 60°/s in free space;
  - 10 mm/s and 15°/s in the slow zone;
  - a step over 3× the limit is refused as a glitch;
  - the gripper may change only once per chunk, and only while the TCP moves ≤ 2 mm in that step.
- Agent 3's scripted demos stay inside these limits: 50 mm/s free, 2.5 mm/s slow, 30°/s.
