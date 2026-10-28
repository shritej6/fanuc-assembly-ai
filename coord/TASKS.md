# Task board

Status: `BACKLOG` → `READY` (set only by the user) → `IN PROGRESS` → `REVIEW` (PR open) → `DONE`.
An agent may only start a task that is READY and owned by it.

| ID | Task | Owner | Status | Branch | Depends on | Day |
|---|---|---|---|---|---|---|
| T01 | Repo scaffold, rules, coord files, contracts v1, vendored official client (`run.py --mock` passes) | CLAUDE | REVIEW | main (initial commit) | none | Mon |
| T02 | Assembly instruction sheets for all 6 products (pictures from CAD, order 01→04, reverse to dismantle) | CLAUDE | READY | feature/T02-assembly-sheets | T01 | Mon |
| T03 | CAD risk ranking v0 (min wall, slender features, layer direction, insertion clearance) → spare/print list | CLAUDE | READY | feature/T03-cad-risk | T01 | Mon |
| T04 | Product catalog + recipe graph from CAD (`assembly/catalog.yaml`, per `contracts/catalog.schema.json`) | CODEX | REVIEW | feature/T04-catalog | T01 | Mon |
| T05 | Adapter layer: `robot/adapter.py` (Sim / Mock / Cell), `policy.py` wiring, UF0 TCP-box guard | CODEX | BACKLOG | feature/T05-adapters | T01 | Mon |
| T06 | CAD defect variants (snapped tab, missing pin/lock, 97/103 % scale, warp, layer split) as STL/USD | CLAUDE | BACKLOG | feature/T06-defect-variants | T03 | Tue |
| T07 | USD assets of all 30 objects + black PLA material; Isaac scene with work area, spare store, reject zone | CLAUDE | BACKLOG | feature/T07-usd-scene | T01 | Tue |
| T08 | Replicator generator + label QA; pilot 1.5k images on Rosie T4 (measure s/img) | CLAUDE | BACKLOG | feature/T08-sdg-pilot | T06, T07 | Tue |
| T09 | Recipe state machine (PackML states), inventory, recovery, Andon; two-independent-check gates | CODEX | BACKLOG | feature/T09-state-machine | T04, T05 | Tue |
| T10 | Full synthetic sets A–D on Rosie T4 (chunks of 500, re-runnable) | CLAUDE | BACKLOG | feature/T10-sdg-full | T08 | Wed |
| T11 | Cosmos Transfer pilot (50 images, measure time + credits) | CLAUDE | BACKLOG | feature/T11-cosmos | T08 | Wed |
| T12 | Isaac Lab Mimic demos for GR00T (after the scripted baseline works in sim) | CLAUDE | BACKLOG | feature/T12-mimic | T09 | Wed |
| T13 | Detector (Faster R-CNN or RT-DETR) + wrist look-alike classifier training/eval | CODEX | BACKLOG | feature/T13-detector | T08 | Wed |
| T14 | FoundationPose setup (STL + depth) + anomalib PatchCore + depth missing-layer check | CODEX | BACKLOG | feature/T14-pose-anomaly | T07 | Wed |
| T15 | Product birth certificate (JSON) + PDF layout | CODEX + CLAUDE | BACKLOG | feature/T15-birth-cert | T09 | Wed |
| T16 | Live OEE + Andon dashboard | CLAUDE + CODEX | BACKLOG | feature/T16-dashboard | T09 | Wed–Thu |
| T17 | Scenario fleet (3 products × 20 layouts × injected faults) + KPIs | CODEX | BACKLOG | feature/T17-fleet | T09, T13 | Thu |
| T18 | GR00T N1.7 / SmolVLA / ACT fine-tune vs the scripted baseline | CODEX | BACKLOG | feature/T18-policy-learn | T12 | Thu |
| T19 | OPC UA PLC-style tags (`asyncua`) | CODEX | BACKLOG | feature/T19-opcua | T09 | Thu |
| T20 | Kanban reorder cards + defect Pareto | CODEX + CLAUDE | BACKLOG | feature/T20-kanban | T09 | Thu |
| T21 | Operator agent: plain-language command → task, live report; Cosmos Reason 2 build check endpoint | CODEX + CLAUDE | BACKLOG | feature/T21-operator | T09 | Thu |
| T22 | `placement.yaml` (10 rollouts with planned faults) + `submission.yaml` | CLAUDE | BACKLOG | feature/T22-placement | T04 | Wed |
| T23 | FAT report from the scenario fleet | CLAUDE | BACKLOG | feature/T23-fat | T17 | Fri |
| T24 | Video script, 1-page summary, references, README | CLAUDE | BACKLOG | feature/T24-submission | all | Sat |

## User-only actions (physical / accounts)

| ID | Action | Status |
|---|---|---|
| U01 | `gh auth login` so the repo can be pushed to GitHub | TODO |
| U02 | Get write access to `/data/ai_club/Innovation_lab_team_v` (ask folder owner `karriv` / Rosie admins to add `gundas` to group `ai_club`) | TODO |
| U03 | Ask organizers: fewer, longer rollouts or back-to-back slots? Real good-part reference photos allowed? | TODO |
| U04 | Print defects + spares (after T03) | TODO |
| U05 | Cycle tests on high-risk parts → results into `assembly/risk_table.json` | TODO |
| U06 | Book real-cell slots on hackathon.sidclab.org (Calendar tab) | TODO |
