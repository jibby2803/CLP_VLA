"""
Training Configuration for LIBERO + Isaac-GR00T
================================================
Centralized configuration file for easy parameter tuning.
"""

from dataclasses import dataclass
from typing import List

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig, ModalityConfig
from gr00t.model.transforms import GR00TTransform


class LiberoDataConfig(BaseDataConfig):
    """Configuration for LIBERO dataset structure and transforms."""
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    def modality_config(self):
        """Define modality configuration for video, state, and action."""
        return {
            "video": ModalityConfig(
                modality_keys=self.video_keys,
                delta_indices=[0],  # Current frame only
            ),
            "state": ModalityConfig(
                modality_keys=self.state_keys,
                delta_indices=[0],  # Current state only
            ),
            "action": ModalityConfig(
                modality_keys=self.action_keys,
                delta_indices=list(range(16)),  # 16 future actions
            ),
            "language": ModalityConfig(
                modality_keys=self.language_keys,
                delta_indices=[0],  # Current language only
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        """Define data transformations for augmentation and preprocessing."""
        transforms = [
            # Video transformations (RGB + depth + mask)
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=256, width=256),
            VideoColorJitter(
                apply_to=["video.image", "video.wrist_image"],  # Only augment RGB
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            
            # State transformations
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            
            # Action transformations
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            
            # Concat transforms - combine video/state/action into single tensors
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            
            # GR00T model-specific transform - creates eagle_* keys
            GR00TTransform(
                state_horizon=1,  # Current state only
                action_horizon=16,  # 16 future actions
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    # Path to LIBERO dataset
    dataset_path: str = "./merged_libero_mask_depth_noops_lerobot_40"
    
    # Video modality keys
    video_keys: List[str] = None
    
    # State keys (robot proprioception)
    state_keys: List[str] = None
    
    # Action keys (robot commands)
    action_keys: List[str] = None
    
    # Video backend
    video_backend: str = "torchvision_av"
    
    # Image resize dimensions
    image_height: int = 256
    image_width: int = 256
    
    # Data augmentation parameters
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.1
    
    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = [
                "video.image",
                "video.wrist_image",
                "video.image_depth",
                "video.wrist_depth",
                "video.image_mask",
                "video.wrist_mask",
                "video.object_of_interest_mask",
                "video.object_of_interest_wrist_mask",
            ]
        
        if self.state_keys is None:
            self.state_keys = [
                "state.x", "state.y", "state.z",
                "state.roll", "state.pitch", "state.yaw",
                "state.gripper",
            ]
        
        if self.action_keys is None:
            self.action_keys = [
                "action.x", "action.y", "action.z",
                "action.roll", "action.pitch", "action.yaw",
                "action.gripper",
            ]


@dataclass
class ModelConfig:
    """Model configuration."""
    # Base model path or HuggingFace ID
    base_model_path: str = "/myproject/GR00T_N1_5_3B/"
    
    # Fine-tuning flags
    tune_llm: bool = False            # Freeze LLM backbone (recommended)
    tune_visual: bool = False          # Freeze visual encoder (recommended)
    tune_projector: bool = True         # Train projector (recommended)
    tune_diffusion_model: bool = True   # Train diffusion model (recommended)
    
    # Compute dtype
    compute_dtype: str = "bfloat16"
    
    # Device
    gpu_id: int = 0  # Which GPU to use (0, 1, 2, ...)

    # Partial Fine-tuning
    tune_visual_layers: int = 0     # Unfreeze last N layers of vision encoder
    tune_llm_layers: int = 0            # Unfreeze last N layers of LLM

    # LoRA Fine-tuning - Modular Configuration
    # =========================================
    # LoRA hyperparameters (shared across modules)
    lora_rank: int = 32                 # LoRA rank (r)
    lora_alpha: int = 32                # LoRA alpha scaling factor
    lora_dropout: float = 0.05          # LoRA dropout
    
    # Per-module LoRA enable flags
    lora_llm: bool = False              # Apply LoRA to LLM backbone (Qwen2)
    lora_vision: bool = False           # Apply LoRA to Vision Encoder (Siglip)
    
    # Note: Action Head and Diffusion Model use full fine-tuning via
    # tune_projector and tune_diffusion_model flags above.
    # LoRA for these components can be added later if needed.
    
    # ====================
    # Geometry Integration (FastVGGT)
    # ====================
    use_geometry: bool = True                                          # Enable geometry mode
    fastvggt_checkpoint: str = "./FastVGGT/model_tracker_fixed_e20.pt"  # FastVGGT checkpoint
    geometry_fusion_mode: str = "both_crossattn"                                 # concat | text_crossattn | vision_crossattn | both_crossattn
    vggt_num_tokens: int = 4                                             # Number of geometry tokens (for concat mode)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Output directory
    output_dir: str = "output/libero_groot_training"
    
    # Training steps
    max_steps: int = 20
    num_train_epochs: int = 300
    
    # Batch size and gradient accumulation
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    learning_rate: float = 2e-4 
    weight_decay: float = 1e-5
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    
    # Mixed precision training
    bf16: bool = True
    tf32: bool = True
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 8
    
    # Logging
    logging_steps: float = 10.0
    report_to: str = "tensorboard"  # Options: "wandb", "tensorboard", "none"
    logging_dir: str = None         # Custom TensorBoard log directory
    
    # DataLoader settings
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = True
    
    # Distributed training
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 100
    
    # Other settings
    seed: int = 42
    do_eval: bool = False
    gradient_checkpointing: bool = False
    torch_compile_mode: str = None  # Options: None, "default", "reduce-overhead", "max-autotune"


# Default configurations
DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()


# Quick configuration presets
QUICK_TEST_CONFIG = TrainingConfig(
    output_dir="output/quick_test",
    max_steps=10,
    per_device_train_batch_size=4,
    save_steps=5,
    logging_steps=1.0,
)

FULL_TRAINING_CONFIG = TrainingConfig(
    output_dir="output/full_training",
    max_steps=180000,
    per_device_train_batch_size=32,
    save_steps=30000,
    logging_steps=10.0,
)

LOW_MEMORY_CONFIG = TrainingConfig(
    output_dir="output/low_memory",
    max_steps=1000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
    gradient_checkpointing=True,
    save_steps=200,
)
