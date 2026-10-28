# T13 Rosie Training Runbook

T13 trains two separate perception models from frozen synthetic manifests:

- RT-DETR sees the 30 designed identities plus the three measured foreign distractor identities.
- The wrist ResNet-18 classifier keeps the exact 30 designed identities and reports look-alike errors separately.

Real-cell images are evaluation-only and must never be placed under the synthetic training root.

## Fixed Rosie paths

The scripts default to these paths and allow explicit environment overrides:

```text
repository       ~/fanuc-assembly-ai
source data      ~/fanuc-assembly-data/synthetic/{A,F,G,B,H,K}
environment      ~/fanuc-t13-venv-secure
prepared data    ~/fanuc-t13/artifacts/{smoke,full}
training output  ~/fanuc-t13/runs/{smoke,full}/{detector,classifier}
```

Set `REPO`, `DATA_ROOT`, `ARTIFACT_ROOT`, and `RUN_ROOT` when the checkout or storage roots differ.

## 1. Create the T4 environment

Run this once on Rosie. The script uses Python 3.12 `venv` and replays the checked-in CUDA 12.6 artifact lock with
`--no-index --require-hashes --only-binary=:all:`. Every dependency is an exact HTTPS wheel URL with SHA-256; the
script disables the pip download cache, verifies installed dependencies, and does not rely on Rosie's broken snap
`uv` executable.

```bash
cd ~/fanuc-assembly-ai
bash perception/rosie/setup_ml.sh
```

The fixed environment uses PyTorch `2.14.1+cu126`, torchvision `0.29.1+cu126`, and Transformers `5.21.0`. These
versions clear the dependency audit used by T13. The T4 nodes' measured driver `610.43.02` supports this runtime;
the V100 nodes' measured driver `525.105.17` does not, so T13 must not be submitted to `dgx` with this environment.

## 2. Run the bounded native smoke gate

Native-only preparation is legal only when `SMOKE=1`. To keep this gate bounded and avoid the known G completion-
metadata mismatch, it validates and freezes the first 128 scenes from detector set A and classifier set B. Those
scenes contain train/validation/test assignments; the trainers select at most 16 training and 8 evaluation samples.
Partial Cosmos files are explicitly excluded from this artifact.

```bash
SMOKE=1 bash perception/rosie/prepare.sbatch

sbatch --parsable --export=ALL,SMOKE=1 perception/rosie/detector.sbatch
sbatch --parsable --export=ALL,SMOKE=1 perception/rosie/classifier.sbatch
```

Record the two job IDs printed by `sbatch`. Do not infer IDs from log filenames.

After each two-step job succeeds, resume its saved checkpoint for exactly one additional step:

```bash
sbatch --parsable --export=ALL,SMOKE=1,MAX_STEPS=3,RESUME_FROM_CHECKPOINT=latest \
  perception/rosie/detector.sbatch
sbatch --parsable --export=ALL,SMOKE=1,MAX_STEPS=3,RESUME_FROM_CHECKPOINT=latest \
  perception/rosie/classifier.sbatch
```

The gate passes only after both resumed runs report finite loss/metrics, readable result JSON, and a measured global
step increase from 2 to 3. Inspect `trainer_state.json`; do not report a step that was not observed.

Use `DRY_RUN=1` to print an exact command without launching training:

```bash
SMOKE=1 DRY_RUN=1 bash perception/rosie/detector.sbatch
```

## 3. Freeze Cosmos-complete production inputs

Wait for T11 to finish importing accepted Cosmos variants. Production preparation has no native-only override: it
rejects any A/F/G/B/H/K set outside the measured 10-20% accepted Cosmos range.

```bash
SMOKE=0 bash perception/rosie/prepare.sbatch
python -m perception.rosie.gates ~/fanuc-t13/artifacts/full/manifests/metadata.json
```

Cosmos images remain train-only and inherit their native scene's split, so neither style variants nor cameras can
leak into validation or test.

## 4. Submit full training

The scripts request one T4 by default. A larger single-node DDP allocation can be requested at submission time;
`torchrun` reads the observed Slurm GPU count rather than assuming hardware that was not allocated.

```bash
sbatch --parsable --gres=gpu:t4:4 --export=ALL,SMOKE=0 perception/rosie/detector.sbatch
sbatch --parsable --gres=gpu:t4:4 --export=ALL,SMOKE=0 perception/rosie/classifier.sbatch
```

To resume an interrupted production run, add `RESUME_FROM_CHECKPOINT=latest`. Resume is rejected when the frozen
manifest, derived COCO/crop metadata, taxonomy, split seed, training configuration, installed package versions,
trainer commit, pretrained model revision, or label mapping differs from the original run. The only permitted
configuration difference is the bounded smoke probe increasing `max_steps` from 2 to 3.

Manifest schema v2 binds every native and Cosmos source image to the SHA-256 of its exact compressed bytes. COCO
records preserve that digest; crop records contain a digest of the exact JPEG bytes written for each crop. Dataset
loaders verify each file from the same in-memory buffer they decode, so a same-path or same-dimension replacement is
rejected before pixels reach either model. The derived metadata hashes therefore transitively bind run and resume
provenance to every consumed image byte.

Production training also requires every frozen detector or classifier label to occur in each train, validation, and
test split; smoke runs skip only this class-completeness gate.

## Outputs to preserve

- `artifacts/full/manifests/metadata.json` and both manifest SHA-256 values
- all COCO JSON and `crops/crops.jsonl`
- each run's `run_provenance.json`, `trainer_state.json`, checkpoints, and measured metric JSON
- observed Slurm job IDs, node/log paths, package versions, and the repository commit

Do not call a run successful because Slurm says `COMPLETED`; verify the metric files and checkpoint/global-step gate.
