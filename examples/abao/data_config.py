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



class VRH31GripperBoth(BaseDataConfig):
    video_keys = ["video.cam_head", "video.cam_left", "video.cam_right"]
    state_keys = [
        # "state.left_arm",
        # "state.right_arm",
        "state.left_shoulder",
        "state.left_elbow",
        "state.left_wrist",
        "state.gripper_open",
        # "state.velocity_left_arm",
        # "state.velocity_right_arm",
        # "state.velocity_left_hand",
        # "state.velocity_right_hand",
        # "state.effort_left_arm",
        # "state.effort_right_arm",
        # "state.effort_left_hand",
        # "state.effort_right_hand",
    ]
    action_keys = [
        # "action.left_arm",
        # "action.right_arm",
        "action.left_arm",
        "action.gripper_open",
        "action.task_progress",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = [x * 2 for x in range(16)]

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
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
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
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
                # target_rotations={
                #     "state.end_effector_rotation_relative": "rotation_6d",
                #     "state.base_rotation": "rotation_6d",
                # },
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
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)