#!/usr/bin/env python3
"""
Inference Server for Custom GR00T Model
=======================================
Hosts the fine-tuned GR00T model as an inference server.
Simulation clients (e.g., evaluate_libero_sim.py) connect to this server to get actions.

Usage:
    CUDA_VISIBLE_DEVICES=1 python serve_custom_policy.py \
        --port 5555
        
    CUDA_VISIBLE_DEVICES=1 python serve_custom_policy.py \
        --port 5556
        
    CUDA_VISIBLE_DEVICES=2 python serve_custom_policy.py \
        --port 5557
        
    CUDA_VISIBLE_DEVICES=2 python serve_custom_policy.py \
        --port 5558
        
    CUDA_VISIBLE_DEVICES=3 python serve_custom_policy.py \
        --port 5559
    
"""

import sys
import os
import tyro
import torch
from dataclasses import dataclass
from typing import Optional

# Add Isaac-GR00T to path
# sys.path.append('./Isaac-GR00T')
sys.path.append('/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom')
#

from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.robot import RobotInferenceServer
from gr00t.data.schema import EmbodimentTag

# Import config
from eval.config import LiberoDataConfig

@dataclass
class ServerConfig:
    """Configuration for inference server."""
    # Path to model checkpoint
    checkpoint: str = "/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom/outputs/2025-12-12/stage2_fully_finetune_from_stage1_v1/checkpoint-50000"
    
    # Port to listen on
    port: int = 5555
    
    # Denoising steps for inference
    denoising_steps: int = 8
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # API Token (optional)
    api_token: Optional[str] = None


def main(cfg: ServerConfig):
    print("=" * 80)
    print("🚀 GR00T Inference Server")
    print("=" * 80)
    print(f"📁 Checkpoint: {cfg.checkpoint}")
    print(f"🔌 Port:       {cfg.port}")
    print(f"🔧 Device:     {cfg.device}")
    
    # 1. Load Policy
    print(f"\n🤖 Loading policy...")
    data_config = LiberoDataConfig()
    
    try:
        policy = Gr00tPolicy(
            model_path=cfg.checkpoint,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            denoising_steps=cfg.denoising_steps,
            device=cfg.device,
        )
        print("✅ Policy loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        sys.exit(1)

    # 2. Start Server
    print(f"\n🌐 Starting server on port {cfg.port}...")
    try:
        RobotInferenceServer.start_server(
            policy=policy,
            port=cfg.port,
            api_token=cfg.api_token
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cfg = tyro.cli(ServerConfig)
    main(cfg)
