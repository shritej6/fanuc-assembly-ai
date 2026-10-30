# Batched cell env (num_envs > 1): API for generators

`AssemblyCellEnv(CellEnvCfg(num_envs=N, ...))` runs N copies of the cell in one Isaac process. Each copy is a full
cell (robot, table, objects, frame markers, cameras, its own lights) under `/World/envs/env_<k>/Cell`, cloned on a 7 m
grid. All copies step together: one `env.step` advances every cell by one 0.1 s action. The GPU step costs the same at
N = 1 and N = 16 (measured, below), so throughput grows about N times. With `num_envs=1`, `reset()` / `step_tcp()`
behave exactly as before.

## Calls

| call | what it does |
|---|---|
| `reset_many([layout_k...], [frames_fn_k...]) -> [obs_k]` | Resets every cell together. All of them settle at once (`mimic_env.settle`). The recorder exports every non-empty episode first, as `env.reset()` does. |
| `begin_reset({k: (layout, frames_fn)})` | **Per-env reset.** Places those cells now. Each one holds home and settles during the next `step_tcp_many` calls while the other cells keep running. Raises if k is already resetting. |
| `ready_resets() -> {k: obs}` | Resets that finished since the last call. The obs is what `reset()` returns (`sim.settled` is False if the 5 s settle cap was hit). Its frames are set, its task is inferred, and its recorder episode starts here: `initial_state` is the settled scene and the settle steps are not recorded. |
| `reset_envs({k: (layout, frames_fn)}) -> {k: obs}` | `begin_reset` plus hold steps until those cells are ready. **Only use it when no other cell is mid-episode**: the hold steps would be recorded into that cell's demo. |
| `step_tcp_many({k: (rows, dt)}) -> {k: obs}` | Plays every listed cell's act_tcp chunk at once, with the same validation and interpolation as `step_tcp`. Unlisted cells, and listed cells whose chunk is shorter, hold their last TCP command and gripper until the longest chunk is done. `{}` makes one hold step, which you need when every live cell is resetting. Raises if a listed cell is still resetting. Resets complete only between calls. |
| `end_episode(k, ok, attrs=None) -> "demo_<n>" or None` | Sets success, exports cell k's episode now and writes `attrs` on its HDF5 demo group. Returns None if there is no recorder, the episode is empty, or the export mode skipped it. Steps cell k records after this are dropped at its next reset. |
| `rotate(next_path) -> closed_path` | Closes the recorder file and records on into `next_path`. Safe at any time: in-progress episodes stay in memory until `end_episode`, which writes them to whichever file is open then. It does **not** annotate; the caller runs agent 3's `annotate_recorder_file(closed_path)`. |

## Observations

Each per-env obs has the same keys as the N = 1 obs:

- `images`, `joints`, `tcp_pose`, `instruction`
- `sim.object_poses`, `sim.contact`, `sim.contact_peak` and `sim.robot_touch` (both over this call's chunk, per env)
- `sim.jaw_n`, `sim.signals`, `sim.signals_latched`, `sim.target`, `sim.base`, `sim.frames`, `sim.tcp_cmd`,
  `sim.clipped_rows`, `sim.on_table`
- `sim.settled`, `sim.settle_s`, `sim.settle_detail`

There is one new key, `sim.ik_error` (str), present when that cell's IK failed. With N > 1 an IK failure does not
raise: that cell holds, its obs carries the error, and its later chunks are ignored until it resets. With N = 1,
`TcpError` is raised as before.

Poses are relative to each cell's own origin (UF0 of that cell), so layouts and frames are the same numbers in every
cell. Recorded states use `scene.get_state(is_relative=True)`. To replay a demo in another cell, call
`reset_to(..., is_relative=True)`.

## Hub rules (agent 7's mimic/batch.py)

1. Call `end_episode(k, ...)` as soon as lane k's episode is decided, before the next `step_tcp_many`. Otherwise the
   hold steps are recorded into its demo.
2. A lane that has its reset obs is live. Wait for its first chunk before stepping again: every step a live cell
   is not listed in is recorded as a hold.
3. While some lanes are resetting, keep calling `step_tcp_many` with the live lanes, or `{}` if there are none. Poll
   `ready_resets()` after every call.
4. Mimic hooks for Isaac Lab's own generator take env_ids:
   - `get_object_poses(env_ids)`, `get_robot_eef_pose(eef, env_ids)` and `get_subtask_term_signals(env_ids)`
     return one row per id.
   - `target_eef_pose_to_action(..., env_id)` returns one row.
   - `set_frames(frames, env_id)`, `object_poses_mm(env_id)`, `frames_mm(env_id)` and `episode_success(env_id)`
     work per env.
   - `env.eps[k]` holds cell k's episode state: task, placements, settle result.

## Measured (fa-g6, 1x L40, product-4 pool, arm held, every contact sensor read each physics step)

| N | env-steps/s, no cameras | env-steps/s, 3 cameras 640x360 |
|---|---|---|
| 1 | 0.918 | 0.705 |
| 4 | 3.739 | 2.916 |
| 8 | 7.378 | 5.826 |
| 16 | 14.766 | 11.657 |

A process step takes about 1.08 s without cameras and 1.37 s with them, at every N. These numbers leave out the
cell env's Python (IK, metrics, signals) and the generator's gates; `batch_check.py` measures the full stack. VRAM
per env and the rates at N = 32 and 64 come from `batch_rate.py`, which records `gpu_mem_used_mib`.
