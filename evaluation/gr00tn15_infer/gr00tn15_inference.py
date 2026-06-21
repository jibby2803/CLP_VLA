import os 
import sys 
import math
import numpy as np 
import torch

# # from lerobot.policies.pi0.modeling_pi0 import PI0Policy
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# from lerobot.policies.factory import make_pre_post_processors

# from gr00t.data.embodiment_tags import EmbodimentTag
# from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

from gr00t.model.policy import Gr00tPolicy
# Import config
from eval.config import LiberoDataConfig
from gr00t.data.schema import EmbodimentTag

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

class Gr00tn15_inference():
    def __init__(self, model_dir="", infer_chunk=10):
        self.model_dir = model_dir 
        self.device = 'cuda'
        self.infer_chunk=infer_chunk
        self.policy = self.create_policy()
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


    def create_policy(self):
        try: 
            
            data_config = LiberoDataConfig()
            print(data_config)
            policy = Gr00tPolicy(
                model_path=self.model_dir,
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                modality_config=data_config.modality_config(),
                modality_transform=data_config.transform(),
                denoising_steps=10, # 8
                device="cuda",
            )
            
            print("Number of parameters: ", sum(p.numel() for p in policy.model.parameters()))
            
            print(f"✅ Loaded N1.5 policy from: {self.model_dir}")
            return policy
        except Exception as e:
            print(f"❌ Failed to load policy: {e}")
            sys.exit(1)


    def get_libero_action(self, obs, task_description):
        data = self._process_observation(obs, task_description)
        try:
            # action_chunk, _ = self.policy.get_action(data)
            action_chunk = self.policy.get_action(data)
        except Exception as e:
            print(f"Error querying server: {e}")
            # Return no-op action on failure
            return np.array(LIBERO_DUMMY_ACTION, dtype=np.float32)
        
        return self.convert_to_libero_action_chunk(action_chunk)

        
    def convert_to_libero_action_chunk(self, action_chunk: dict, start_idx: int = 0):
        actions = []
        # action_chunk = self.post_process_action_chunk(action_chunk)
        for t in range(start_idx, start_idx + self.infer_chunk):
            action_components = []
            for key in self.action_keys:
                val = action_chunk.get(f"action.{key}")[0]
                # print(val.shape)
                if val is None:
                    raise ValueError(f"Missing key action.{key} in server response")

                # Handle per-timestep selection. Support scalars and arrays.
                if hasattr(val, "shape") and len(val.shape) > 0:
                    # Protect against out-of-range idx
                    if t >= val.shape[0]:
                        raise IndexError(f"Requested idx {t} for key {key}, but available length is {val.shape[0]}")
                    val_t = val[t]
                else:
                    # Scalar per key; same value for all timesteps
                    val_t = val

                # If the timestep value is vector-valued, extend; if scalar, append
                val_t = np.array(val_t, dtype=np.float32).reshape(-1)
                action_components.extend(val_t.tolist())

            action_array = np.array(action_components, dtype=np.float32)
            action_array = self.normalize_gripper_action(action_array, binarize=True)  # your existing normalization
            action_array = invert_gripper_action(action_array)
            actions.append(action_array)
        return np.stack(actions, axis=0)  # shape: (10, D)
    
    
    def _process_observation(self, obs, task_description):
    
        xyz = obs["robot0_eef_pos"]
        rpy = _quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img = obs["agentview_image"][::-1, ::-1]
        wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        
        # Ensure images are uint8
        img = img.astype(np.uint8)
        wrist_img = wrist_img.astype(np.uint8)
        
        # GR00T N1.6 expects:
        # - video: shape (B, T, H, W, C) - add batch and temporal dims
        # - state: shape (B, T, D) - add batch and temporal dims
        # - language: tuple of strings (B,)
        new_obs = {
            # Video: (H, W, C) -> (1, 1, H, W, C) for B=1, T=1
            "video.image": img[np.newaxis, np.newaxis, ...],  # (1, 1, 256, 256, 3)
            "video.wrist_image": wrist_img[np.newaxis, np.newaxis, ...],  # (1, 1, 256, 256, 3)
            # State: scalar -> (1, 1, 1) for B=1, T=1, D=1
            "state.x": np.array([[[xyz[0]]]], dtype=np.float32),  # (1, 1, 1)
            "state.y": np.array([[[xyz[1]]]], dtype=np.float32),  # (1, 1, 1)
            "state.z": np.array([[[xyz[2]]]], dtype=np.float32),  # (1, 1, 1)
            "state.roll": np.array([[[rpy[0]]]], dtype=np.float32),  # (1, 1, 1)
            "state.pitch": np.array([[[rpy[1]]]], dtype=np.float32),  # (1, 1, 1)
            "state.yaw": np.array([[[rpy[2]]]], dtype=np.float32),  # (1, 1, 1)
            # Gripper: (2,) -> (1, 1, 2) for B=1, T=1, D=2
            "state.gripper": gripper.astype(np.float32)[np.newaxis, np.newaxis, ...],  # (1, 1, 2)
            # Language: tuple of strings with B elements
            "annotation.human.action.task_description": (str(task_description),),  # tuple of 1 string
        }
        return new_obs

    def normalize_gripper_action(self, action, binarize=True):
        orig_low, orig_high = 0.0, 1.0
        action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

        if binarize:
            # Binarize to -1 or +1.
            action[..., -1] = np.sign(action[..., -1])

        return action


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action