# LeRobot export for GR00T (Agent 4 → Agent 5)

`policy_learn/groot/prepare_dataset.py` reads this format, checks every rule below, and stops with the full list of
problems if a rule fails. It never changes your files. Put each dataset in its own folder:
`~/t12_data/lerobot/<dataset>/` (for example one per skill). The trainer uses every folder that has
`meta/info.json`, and folder names must be unique.

The format is plain **LeRobot v2.1**, the version GR00T N1.7 reads. If your exporter writes v3.0, convert it with
Isaac-GR00T's `scripts/lerobot_conversion/convert_v3_to_v2.py`. Do not write `meta/modality.json`; the prepare step
writes it.

## Files

```
<dataset>/
  meta/info.json        codebase_version "v2.1", fps 5, chunks_size 1000, data_path / video_path as below, features
  meta/episodes.jsonl   {"episode_index": i, "tasks": ["<instruction>"], "length": n}, one line per episode
  meta/tasks.jsonl      {"task_index": k, "task": "<instruction>"}
  data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
  videos/chunk-{episode_chunk:03d}/observation.images.{table|side|wrist}/episode_{episode_index:06d}.mp4
```

- Episodes are numbered 0..N-1 with no gaps, and chunk = episode_index // 1000.
- Keep only the demos that passed the force guard and the success check.

## Parquet columns (one row per control step)

| Column | Type | Content |
|---|---|---|
| `observation.state` | float32[13] | `[J1..J6 deg, gripper, x, y, z, w, p, r]`: the observation's `joints` (7) followed by `tcp_pose` (6), read **before** the action of this row is sent |
| `action` | float32[7] | the `act_tcp` row sent at this step: `[x, y, z, w, p, r, gripper]`. These are absolute UF0 targets in mm and degrees, gripper 0 open / 1 closed, exactly what `CellClient.act_tcp` gets |
| `timestamp` | float32 | `frame_index / 5` |
| `frame_index` | int64 | 0..n-1 within the episode |
| `episode_index` | int64 | the episode number |
| `index` | int64 | the global row number across the dataset |
| `task_index` | int64 | the row of `tasks.jsonl` that holds this episode's instruction |
| `next.done` | bool | optional; true on the last row |

Rules that are checked:
- **The rate is 5 Hz.** One row is one 0.2 s runtime step (`policy_learn/deploy/groot_policy.py` `DT_S`, the real
  cell's act dt). The recorder writes one row per 0.1 s Isaac env step (`sim/assembly_lab/skills/motion.py` `DT_S`);
  `export_lerobot.py` keeps every 2nd row and refuses files that state another row period. Do not resample further.
- `action[t]` is the target the arm moves to during step t. On the last row, repeat the final target (hold).
- Every `action` xyz stays inside the workspace box (x 150–650, y −450–450, z −334–300 mm, ±1 mm).
- The gripper value in both state and action is between 0 and 1. Every demo uses exactly 0 or 1.
- WPR follow the FANUC fixed-angle convention, R = Rz(r)·Ry(p)·Rx(w). Fingers straight down are w, p, r =
  −180, 0, r. Either sign of ±180 for w is fine, because the prepare step converts to a rotation matrix.
- There are no NaN or inf values, and the parquet length equals `length` in `episodes.jsonl`.
- **The instruction** in `tasks.jsonl` is the planner's text that GR00T will also get at run time, with the plan's
  target in it. Example: `pick M4_02 at (412.0, -87.5, yaw 33) gripping X, insert on F04 at (300.0, 120.0, yaw 90)`.
  - Use exactly the template that the runtime planner prints, with the same number format.
  - Thousands of distinct instructions are fine.
- **Do not include** the sim-only extras (`object_poses`, `contact`) in any column. A policy never reads them.

## Videos

- There are 3 cameras: `table`, `side`, `wrist`. Each one is an MP4 file with **H.264**, yuv420p and 5 fps. Do not
  use AV1, which GR00T's torchcodec decoder does not support reliably.
- Each frame is **640×360**: the whole 16:9 camera frame, resized and not cropped. This is the same image the robot
  policy gets (`robot/cell/policy.py` `IMAGE_SIZE`).
- The frame count equals the episode length, and frame t is the image at the same moment as `observation.state[t]`.
- In `info.json`, each video feature is `{"dtype": "video", "shape": [360, 640, 3], "names": ["height", "width",
  "channels"], ...}`.

## What the prepare step produces (for reference)

The prepare step writes `~/groot_data/train/<dataset>` plus `~/groot_data/val/<dataset>`. The val copy holds 2 % of
the episodes (at least 1 and at most 50) back from training for the open-loop eval. The prepare step also:
- adds `observation.eef_9d` and `action.eef_9d`, which are `[x, y, z mm, rot6d]` with rot6d = the first two rows
  of R. This is GR00T's continuous EEF format. Raw WPR jumps by 360° at w = ±180, which a regression head cannot
  learn across.
- writes `modality.json` for `fanuc_er4ia_config.py`:
  - state: `eef_9d`, `gripper`, `joints`
  - action: `eef_9d`, `gripper` (16 rows = 3.2 s)
  - video: `table`, `side`, `wrist`
  - language: `annotation.human.task_description`
- links the videos instead of copying them.

GR00T's `gr00t/data/stats.py` then writes `meta/stats.json` and `meta/relative_stats.json`, the normalisation
statistics (q01/q99 min-max to [−1, 1]).
