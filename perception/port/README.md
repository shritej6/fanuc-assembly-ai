# T30 perception port

`CellPerceptionPort` implements the T09 `PerceptionPort` Protocol (`assembly/controller_ports.py`): `scan`,
`pre_pick`, `post_place`, `post_transfer`, `collision_scene`. Every result carries the observation's own `t`. When the
port cannot produce trustworthy evidence it raises a `PerceptionPortError`, which is a `ControllerPortError`, so the
controller holds. It does not return a guess.

## How the controller constructs it

```python
from pathlib import Path
from perception.port import CellPerceptionPort
from perception.port.backends import RtDetrDetector, DefectHeadClassifier
from perception.port.geometry import CalibratedCellGeometry
from perception.port.judge import CloseupThresholds
from perception.port.placement import PoseUncertainty, precision_gate_tolerances
from perception.port.scene import ArmCorridor
from perception.port.size_gate import WRIST_150MM_PRELIMINARY

port = CellPerceptionPort(
    detector=RtDetrDetector(Path("<run>/detector"), min_score=...),     # T27/T31 RT-DETR r50vd @1280
    classifier=DefectHeadClassifier(Path("<run>/classifier"), labels, image_size),
    pose_estimator=<FoundationPose service>,                          # PoseEstimator protocol, see "Blocked"
    geometry=CalibratedCellGeometry(),                                # robot/cell/calibration + T28 localize
    thresholds=CloseupThresholds.load(Path("<measured thresholds>.json")),
    wrist_size=WRIST_150MM_PRELIMINARY,
    wrist_pose=PoseUncertainty(..., status=..., source=...),          # FoundationPose bound, wrist
    table_pose=PoseUncertainty(..., status=..., source=...),          # FoundationPose bound, table
    arm_corridor=ArmCorridor(radius_mm=..., source=...),
    image_sink=<callable(name, rgb) -> ref>,                          # stores the Andon pictures
)
execution_tolerances, perception_tolerances = precision_gate_tolerances()   # 0.3 mm / 1 deg CAD gates
```

The policy must request depth for both the table and the wrist cameras (`DEPTH = ("table", "wrist")`).

**No defaults for unmeasured values.** The thresholds, the FoundationPose bounds and the arm corridor have no
default. Each one must name its source. `WRIST_150MM_PRELIMINARY` and `TABLE_DEPTH_PRELIMINARY` carry their status
and source.

## Models and weights

| Role | Model | Weights |
|---|---|---|
| Table and side detector | RT-DETR r50vd at 1280 (`perception/cell/models.Detector`) | Rosie `~/fanuc-t31-results/fanuc-t27-runs/main/full/detector` |
| Wrist identity + good/defect | ResNet-18 with the defect head, 32 logits (`classifier_defect_v100`) | T27 model `5696b36c` (job 291308). The loader needs `_build_defect_model` from `feature/T27-newset-training` `d869bad`, which is not merged here |
| 6D pose | FoundationPose against the printed STL | none: it is pretrained. It runs in the `fa-foundationpose` container on CUDA |

## What each call returns

- **`scan(obs) -> InventorySnapshot`**
  - Pipeline: table RT-DETR boxes → class-agnostic suppression (`dedup.py`: IoU ≥ 0.5, or ≥ 0.85 of the smaller box
    covered; top-1 label wins, runner-ups kept) → UF0 centre from the table depth (T28 `locate`) → zone → T28
    tracker → `Ledger`.
  - Ids: the ledger keeps each published id stable. If a track's identity changes, the ledger mints a new id
    (`p7r1`), because the inventory refuses identity flips. When the robot moves a part, its id follows it.
  - Item conditions:
    - `__fallen` → `FALLEN`;
    - label vote share under 0.5, or a runner-up of another shape within 0.8 of the winner's score → `UNCERTAIN`;
    - otherwise `VERIFIED`.
  - A base that carries installed parts is reported as observed while something stands on its footprint.
  - Typed extras in `port.last_scan` (`ScanReport`): foreign and unknown pieces (the inventory accepts only the 30
    catalog ids), pieces outside the work area, boxes with no depth, side-camera foreign boxes, and `needs_closeup`.
  - Table-depth size: it is reported, and it is not used to downgrade a part. Its xy bound (6.42 mm) is assumed.
  - FoundationPose runs on every base, because the base pose feeds `collision_scene` and the seat check.
- **`pre_pick(obs, part) -> EvidencePair`**
  - The close-up is a wrist view at the inspection stop.
  - Part depth points: the points near the scan centre, in the largest connected group.
  - FoundationPose on those points. The STL pose is converted to installed-local with
    `world_from_stl @ inv(T_local_from_stl)`, the same composition the controller uses.
  - The posed CAD box is projected into the wrist image. The classifier gives identity and `p_defect`.
  - **Size gate** (`size_gate.py`): it measures the part first, then matches the size to the CAD:
    - footprint (sorted extents in the part frame);
    - height (median of the top-20 % of 2 mm cell medians);
    - CAD top cues where a look-alike pair has one, for example the M1_01 pocket floor at 7 mm against the solid
      M3_01 at 16 mm, or the M1_02 post at 17 mm against M2_02 at 16 mm.
  - The gate checks against the part's look-alikes, its size twins and all 30 objects. Outcomes: `confirms`,
    `cannot_separate`, `contradicts`, `matches_nothing` or `unmeasured`.
  - Verdict rules are in `judge.py`, in order:
    1. fit below its minimum → unsure;
    2. fallen → `flipped`/reject;
    3. fits none of the 30 and the classifier is unsure → `foreign`/reject;
    4. `p_defect` ≥ 0.994 → `print_defect`/reject;
    5. classifier below its minimum → unsure;
    6. another shape confirmed by size → `wrong_part`/reject, otherwise unsure;
    7. size contradicts the expected part, fits nothing, or has no depth → unsure;
    8. `p_defect` in [0.5, 0.994) → unsure;
    9. otherwise `ok`/use.
  - F01 = F02 and F04 = F05 = F06 are one mesh, so any base of the type counts as the expected base.
  - Unsure becomes `uncertain`/`ask_operator` with `checks_agree` false, which the controller turns into HOLD_ANDON.
  - **`port.closeup(id)`** returns a `CloseupReport` with `next_step` set to `use`, `reject`, `second_look` or
    `andon`:
    - `second_look` carries a typed `ViewRequest`: 150 mm standoff, 30° from vertical, from the part side that
      LOOKALIKES.md names. The controller owns turning it into a safety-approved motion.
    - A view counts as the second look only when the tool w or p differs by at least 15° from the first view.
    - If the second look is still unsure, `andon` lists both image refs.
- **`post_place(obs, part) -> PlacementEvidence`**
  - The part is still in the fingers, so the size and defect checks are not repeated.
  - Checks: identity (classifier + fit), seat and tilt.
  - Seat error = (FoundationPose z of the part's installed origin) − (measured base origin z) − `seated_origin_z_mm`.
    The gate is ±0.3 mm. Tilt is gated at 1°. Both gates come from INTERFACES.md.
  - A gate is **certified** only with a MEASURED bound and |error| + bound ≤ gate. It is **refuted** when
    |error| − bound > gate. Anything in between is unsure.
  - Certified → `seated` = `pose_agrees` = true. Refuted → both false (the controller lifts 3 mm and reinserts
    once). Unsure → `uncertain`/`ask_operator`, which becomes HOLD_ANDON.
  - `port.placement(id)` returns the measured errors, the bound and its status.
- **`post_transfer(obs, part, zone) -> RecoveryTransferEvidence`**
  - A wrist look at the released part. `pose_agrees` is the calibrated zone check (REJECT, SPARE slot, or STAGED on
    the test bed).
- **`collision_scene(obs, moving, base) -> CollisionScene`**
  - The base pose is re-fitted with FoundationPose on this frame.
  - Fixtures: posed CAD AABBs of the bases.
  - Parts: CAD-footprint squares at the scanned centres. Foreign pieces use their measured footprint. The moving part
    is excluded.
  - `moving_attached`: the gripper is closed and wrist depth points lie in the slab below the fingertips.
  - `intrusion_detected`: table-depth points taller than the tallest CAD stack, outside the arm corridor.

## Measured numbers (real captures `~/fanuc-cell-captures/r20261029-995508`, evaluation only, never training)

- **Duplicate suppression** (stored detections at score ≥ 0.5)
  - Reference: the visual count in `detector_check_new/visual_count_agent.json`. It holds agent-drawn object boxes,
    not human ground truth.
  - A box is counted against an object when its centre lies in the object box.

  | detector | cameras | objects | duplicates before → after | objects with no box before → after |
  |---|---|---|---|---|
  | new (T31) | table + side | 412 | 125 → 36 | 24 → 24 |
  | new (T31) | wrist | 224 | 18 → 2 | 78 → 78 |
  | old (T27) | table + side | 412 | 91 → 35 | 19 → 19 |

  Suppression removed 71 % of the T31 table/side duplicates and lost no object. With the containment rule, 5 fewer
  boxes were kept than with IoU alone (511 → 506), and duplicates fell by one more (37 → 36). The remaining 36 are
  boxes that do not overlap enough, such as part-only boxes inside an object's region.
- **Wrist depth noise**: in 80 40×40 px patches of 16 frames at a median range of 364 mm, the residual from a local
  plane fit had:
  - std median 7.64 mm (p10–p90 5.32–11.93);
  - robust sigma median 6.54 mm.

  This is about 4× the D405-class theory (1.7 mm at 400 mm in LOOKALIKES.md). Scaled by range squared to 150 mm it
  gives **1.3 mm (PRELIMINARY extrapolation)**, and that is the z bound in `WRIST_150MM_PRELIMINARY`. At that bound
  the 1 mm M1_02/M2_02 height difference **cannot** be separated by depth (a test records this). The wrist classifier
  decides those parts.

## Known limits

- **±0.3 mm seat**: it is not certifiable from the camera today. No MEASURED FoundationPose bound exists for either
  camera. With a PRELIMINARY bound, every placement that is not clearly wrong goes to the operator.
  - To unblock: measure FoundationPose repeatability on a seated part at the post-place pose (10 frames, report
    std).
  - Then pass `PoseUncertainty(..., Status.MEASURED, source)`.
- **Depth on black PLA at 150 mm**: unmeasured. A depth hole makes the part "unmeasured", which counts as unsure.
- **Table-depth size**: 3.2 mm pixels cannot separate 1–2 mm look-alikes. It only flags gross mismatches.
- **Post-place mask**: it can include the housing walls around a seated part, because points above the expected part
  bottom are kept. Fingers hide part of the top, so post-place uses FoundationPose z and not a top-surface height.
- **Held-part check**: near a base, the base can enter the slab, so this is a drop detector for carries.
- **Collision parts**: positions come from the last scan. Parts move only when the robot moves them (T28 tracker
  assumption). Bases and the held part are measured on every call.
- **Shared tolerance**: the same `PerceptionTolerances` (0.3 mm) also gates base drift in `collision_scene`. A base
  fit noisier than that holds the cell.
- **Dedup parameters**: IoU 0.5, containment 0.85 and contested ratio 0.8 are chosen, not fitted.
- **Second-look side**: the part-local side convention (front = −y) is inferred from the base front L-stops at local
  y −16.5.

## Blocked (who)

| What | On whom |
|---|---|
| FoundationPose service implementing `PoseEstimator` (CUDA container on the RunPod A100 / Rosie) | T28 / main |
| Measured `CloseupThresholds` (classifier min, fit min) for the deployed models | T27 / T28 |
| Defect-head loader code merged (`feature/T27-newset-training` `d869bad`) | main (merge) |
| Measured FoundationPose bounds (wrist, table) to certify the 0.3 mm / 1° gates | first real-cell minutes (main) |
| Sourced arm-corridor radius or a robot-link model for intrusion | T09 / Codex (robot-link provider) |
| Controller motion for the second look (`ViewRequest` → safety-approved TCP) and an Andon hook | T09 / Codex |

## Proposed contract / Protocol changes (doc only; `contracts/` and `assembly/` are unchanged)

1. `inspection.evidence.defect_score: number | null`, the defect head's `p_defect`. This is the T31 proposal in
   DEFECT_HEAD_INSPECTION.md.
2. `PerceptionPort.scan` should be able to return foreign or unknown pieces, for example
   `InventorySnapshot.foreign: tuple[ForeignObject, ...]`, so the controller can move them to the defect area
   (ROBOT_BRAIN §3). Today they are only obstacles plus `ScanReport.foreign`.
3. `EvidencePair` should carry a typed `next_step` (`second_look` with a `ViewRequest`, or `andon` with image refs),
   so the controller reads `port.closeup(id)` without a side channel.
4. `PlacementEvidence` should carry the measured seat and tilt errors and their bound status. Then "unsure"
   (operator) and "refuted" (recover) do not depend on `seated` / `pose_agrees` booleans.
