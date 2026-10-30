"""GR00T N1.7 modality config: FANUC ER-4iA + Team V gripper (registered as NEW_EMBODIMENT).

Loaded by Isaac-GR00T's `launch_finetune.py --modality-config-path` and `gr00t/data/stats.py`; it imports only
`gr00t`. The keys match the `meta/modality.json` that `prepare_dataset.py` writes (see DATA_FORMAT.md).

Layout mirrors GR00T's DROID pre-training embodiment (single arm, exterior + wrist cameras, eef_9d):
- video  table, side, wrist (current frame)
- state  eef_9d (TCP xyz mm + rot6d), gripper (0 open / 1 closed), joints (J1..J6 deg)
- action eef_9d + gripper, ACTION_HORIZON rows at the dataset fps (5 Hz = the demos' dt 0.2 s)
- language annotation.human.task_description (the planner's instruction)

The dataset stores ABSOLUTE act_tcp targets and the policy returns ABSOLUTE targets either way. GROOT_EEF_REP
picks what the network regresses: "relative" (default; the N1.7-recommended relative EEF, delta from the current
TCP, converted back to absolute by GR00T's processor with the current state) or "absolute".
Changing GROOT_ACTION_HORIZON or GROOT_EEF_REP needs fresh stats (gr00t/data/stats.py fingerprints them).
The values used are saved into the checkpoint, so serving never re-reads these env vars.
"""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = int(os.environ.get("GROOT_ACTION_HORIZON", "16"))
EEF_REP = os.environ.get("GROOT_EEF_REP", "relative").lower()
if EEF_REP not in ("relative", "absolute") or not 1 <= ACTION_HORIZON <= 40:
    raise ValueError(f"GROOT_EEF_REP={EEF_REP!r} must be relative|absolute; GROOT_ACTION_HORIZON 1..40")
LANGUAGE_KEY = "annotation.human.task_description"

fanuc_er4ia_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["table", "side", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["eef_9d", "gripper", "joints"]),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["eef_9d", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE if EEF_REP == "relative" else ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_9d",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper",
            ),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
}

register_modality_config(fanuc_er4ia_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
