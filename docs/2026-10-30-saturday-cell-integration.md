# Saturday cell integration plan

Owner: CODEX  
User authorization: 2026-10-30  
Deadline: 2026-10-30 22:00 America/Chicago

## Outcome

Produce one reviewed integration branch where `robot/cell/policy.py` can opt into
`runtime="cell"`, constructs the T09 controller, and routes every real pick,
insert, and pull-out motion through `policy_learn.deploy.GrootPolicy` and its
deployment guards. There is no scripted real-cell motion fallback.

## Binding constraints

- Preserve the official `robot/cell/cell_client.py` byte-for-byte.
- Default startup remains fail-closed; cell mode requires explicit configuration.
- Perception proposes poses; controller plans and verifies; GR00T owns real motion;
  deployment guards authorize every emitted chunk.
- Consume T29 turn-indexed fingertip TCPs. Never command the deprecated pad point.
- Use only measured or provenance-pinned limits. Missing inputs cause Held/Andon.
- Keep PRs #7, #8, and #18 blocked.
- Defer #10-#13, #16, and #17 until Sunday.

## Dependency order

1. #6 error-code contract
2. #19 look-alike contract
3. #4 T05 adapters
4. #15 T29 grasp frames
5. #5 T04 catalog, repinned to final T29
6. #9 T09 controller, updated for final catalog/T29 boundaries
7. #21 T30 PerceptionPort
8. T12 deploy adapter and GR00T-only cell wiring
9. #22 Cosmos Reason 2 8B pin

## Test contract

- Regression tests are written and observed failing before production changes.
- Targeted tests pass after each integration boundary.
- Final gates: full `uv run pytest -q`, `uv run ruff check .`,
  `git diff --check`, protected-client hash check, official mock run, and an
  explicit fail-closed cell-runtime test without a live endpoint.
- A fresh reviewer audits the complete final diff before the branch is reported.

## Out of scope tonight

- Physical real-cell motion or pendant teaching.
- Merging PRs on the user's behalf.
- Dashboards, OPC UA, Kanban, certificates, scenario fleet, or the older runtime.
- Weak/incomplete training monolith, unsafe booking package, or incomplete T14 run.
