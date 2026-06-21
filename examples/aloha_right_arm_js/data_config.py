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
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class RightArmAlohaDataConfig(BaseDataConfig):
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]

    state_keys = [
        "state.right_waist",
        "state.right_shoulder",
        "state.right_elbow",
        "state.right_forearm_roll",
        "state.right_wrist_angle",
        "state.right_wrist_rotate",
        "state.right_gripper"
    ]

    action_keys = [
        "action.right_waist",
        "action.right_shoulder",
        "action.right_elbow",
        "action.right_forearm_roll",
        "action.right_wrist_angle",
        "action.right_wrist_rotate",
        "action.right_gripper",
    ]

    language_keys = [
        "annotation.human.action.task_description",
    ]
    observation_indices = [0]
    action_indices = list(range(16)) #list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:

        action_transform = StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={key: "min_max" for key in self.action_keys},
        )

        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
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
            action_transform,
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RightArmAlohaDataConfigMeanStd(RightArmAlohaDataConfig):
    """Apply mean_std normalization to actions other than gripper."""

    def transform(self) -> ModalityTransform:
        return super().transform(action_norm="mean_std")