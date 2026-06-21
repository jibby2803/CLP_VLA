# "vr_h311_left_arm_quant2_include_task_progress_exclude_velocity_effort": VRH311LeftArmQuant2IncludeTaskProgressExcludeVelocityEffortConfig(),
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.data.dataset import ModalityConfig

from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class VRH311LeftArmQuant2IncludeTaskProgressExcludeVelocityEffortConfig(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.left_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(0, 32, 2))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.state_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)