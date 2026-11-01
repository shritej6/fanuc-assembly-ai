"""GR00T N1.7 modality config for the FANUC ER-4iA with the team gripper.
Authored by Shritej Gunda."""

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
