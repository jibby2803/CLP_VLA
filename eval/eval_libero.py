#!/usr/bin/env python3
"""
Simulation Evaluation Client for LIBERO
=======================================
Connects to the GR00T inference server and runs evaluation on LIBERO tasks.
Calculates success rate and saves rollout videos.

Usage:
    python eval_debug.py \
        --exp-name base_scale100_bz24_lr2e-4_40k \
        --task-suite-name libero_object
        
    python eval_all.py \
        --exp-name base_scale40_bz24_lr2e-4_40k
"""

import sys
import os
import tyro
import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Add Isaac-GR00T to path
# sys.path.append('/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom')
sys.path.append('/mnt/data/sftp/data/vla_intern/workspace/binh/gr00t_custom')
# sys.path.append('./LIBERO')

from libero.libero import benchmark
from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
# from gr00t.eval.service import ExternalRobotInferenceClient
from gr00t.model.policy import Gr00tPolicy
# Import config
from eval.config import LiberoDataConfig
from gr00t.data.schema import EmbodimentTag


from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import torch.multiprocessing as mp

@dataclass
class SimConfig:
    """Configuration for simulation evaluation."""
    # Task suite name (e.g., libero_spatial, libero_object, libero_goal, libero_10, libero_90)
    exp_name: str = 'test'
    
    model_dir: str = '/mnt/data/sftp/data/vla_intern/workspace/binh/gr00t_custom/outputs/2025-12-20/base_scale40_lr2e-4/checkpoint-40000'
    # model_dir: str = '/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom/outputs/2025-12-16/base_scale40_1gpu_bz24/checkpoint-40000'
    
    task_suite_name: str = "libero_spatial"
    
    
    # Number of trials per task
    num_trials_per_task: int = 50
    
    # Server host
    host: str = "localhost"
    
    # Server port
    port: int = 5555
    
    # Number of steps to wait for objects to stabilize
    num_steps_wait: int = 10
    
    # Output directory for logs and videos
    output_dir: str = "./eval_libero_results"
    
    # Headless mode (no GUI)
    headless: bool = True


class GR00TClientPolicy:
    """Client-side policy wrapper that communicates with the inference server."""
    
    # def __init__(self, host="localhost", port=5555, model_dir=""):
    def __init__(self, model_dir=""):
        # self.client = ExternalRobotInferenceClient(host=host, port=port)
        self.model_dir = model_dir
        self.client = self.create_client()
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    def get_action(self, obs_dict, lang: str):
        """Get action from server given observation and language instruction."""
        # Process observation to match GR00T format
        processed_obs = self._process_observation(obs_dict, lang)
        
        # Query server
        try:
            action_chunk = self.client.get_action(processed_obs)
        except Exception as e:
            print(f"Error querying server: {e}")
            # Return no-op action on failure
            return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            
        # Convert to LIBERO format
        # return self._convert_to_libero_action(action_chunk, 0)
        return self.convert_to_libero_action_chunk(action_chunk)
    
    def create_client(self):
        
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
            return policy
        except Exception as e:
            print(f"❌ Failed to load policy: {e}")
            sys.exit(1)
        

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        return new_obs

    def _convert_to_libero_action(self, action_chunk: dict, idx: int = 0) -> np.ndarray:
        """Convert GR00T action chunk to Libero format."""
        # Check if action keys exist in response
        if "action.x" not in action_chunk:
             # Handle case where server might return 'actions' key or similar if structure differs
             # But based on service.py, it returns what the model returns.
             # If model returns dict with 'action.x', etc., we are good.
             pass

        action_components = []
        for key in self.action_keys:
            # Handle both scalar and array returns
            val = action_chunk.get(f"action.{key}")
            if val is None:
                raise ValueError(f"Missing key action.{key} in server response")
            
            # Extract specific time step
            if hasattr(val, "shape") and len(val.shape) > 0:
                val = val[idx]
            
            action_components.append(np.atleast_1d(val)[0])
            
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        return action_array
    
    
    # import numpy as np

    def convert_to_libero_action_chunk(self, action_chunk: dict, start_idx: int = 0, n_action: int = 10) -> np.ndarray:
        """
        Extract 10 actions starting at start_idx, using the same per-timestep logic as _convert_to_libero_action.
        Returns shape: (10, num_components)
        """
        actions = []
        for t in range(start_idx, start_idx + n_action):
            action_components = []
            for key in self.action_keys:
                val = action_chunk.get(f"action.{key}")
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
            action_array = normalize_gripper_action(action_array, binarize=True)  # your existing normalization
            actions.append(action_array)

        return np.stack(actions, axis=0)  # shape: (10, D)



def main(cfg: SimConfig, task_suite_name: str=None):
    print("=" * 80)
    print("🎯 LIBERO Simulation Evaluation")
    print("=" * 80)
    if task_suite_name is not None:
        cfg.task_suite_name = task_suite_name
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Trials/Task: {cfg.num_trials_per_task}")
    print(f"Server: {cfg.host}:{cfg.port}")
    
    
    # Create output directory
    os.makedirs(cfg.output_dir + '/' + cfg.exp_name, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, cfg.exp_name, f"eval_{cfg.task_suite_name}_12.log")
    log_file = open(log_path, "w")
    log_file.write(f"Checkpoint path: {cfg.model_dir}")
    # Initialize Policy Client
    print("\n🔌 Connecting to inference server...")
    try:
        # policy = GR00TClientPolicy(host=cfg.host, port=cfg.port)
        policy = GR00TClientPolicy(model_dir=cfg.model_dir)
        # Simple ping to check connection
        # policy.client.ping()
        print("✅ Connected to server!")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("   Make sure 'serve_custom_policy.py' is running!")
        sys.exit(1)

    # Initialize LIBERO Benchmark
    print("\n📦 Initializing LIBERO benchmark...")
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        num_tasks = task_suite.n_tasks
        print(f"✅ Loaded {num_tasks} tasks from {cfg.task_suite_name}")
    except Exception as e:
        print(f"❌ Failed to load benchmark: {e}")
        sys.exit(1)

    # Evaluation Loop
    total_episodes = 0
    total_successes = 0
    
    print("\n🚀 Starting evaluation...")
    
    # tasks_ids = [0, 1]
    # for task_id in tasks_ids:
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, resolution=256)
        
        print(f"\nTask {task_id+1}/{num_tasks}: {task_desc}")
        log_file.write(f"\nTask {task_id+1}: {task_desc}\n")
        
        task_successes = 0
        
        # Determine max steps based on suite
        if cfg.task_suite_name == "libero_spatial": max_steps = 220
        elif cfg.task_suite_name == "libero_object": max_steps = 280
        elif cfg.task_suite_name == "libero_goal": max_steps = 300
        elif cfg.task_suite_name == "libero_10": max_steps = 520
        elif cfg.task_suite_name == "libero_90": max_steps = 400
        else: max_steps = 300
        
        for trial in range(cfg.num_trials_per_task):
            env.reset()
            # Set initial state
            init_state_idx = trial % len(initial_states)
            env.set_init_state(initial_states[init_state_idx])
            
            # Rollout
            t = 0
            done = False
            top_view_frames = []
            wrist_view_frames = []
            
            print(f"   Trial {trial+1}/{cfg.num_trials_per_task}...", end="", flush=True)
            log_file.write(f"   Trial {trial+1}/{cfg.num_trials_per_task}...")
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait for stabilization
                    if t < cfg.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    
                    # Get images for video
                    img, wrist_img = get_libero_image(obs)
                    top_view_frames.append(img)
                    wrist_view_frames.append(wrist_img)
                    
                    # Get action
                    action = policy.get_action(obs, task.language)
                    
                    # Step environment
                    for act in action:
                        obs, reward, done, info = env.step(act.tolist())
                    # obs, reward, done, info = env.step(action.tolist())
                    t += 1
                    if done:
                        break
                        
                    
                except Exception as e:
                    print(f" Error: {e}")
                    break
            
            # Record result
            success = done
            if success:
                task_successes += 1
                total_successes += 1
                print(" ✅ Success")
                log_file.write("✅ Success")
            else:
                print(" ❌ Failure")
                log_file.write(" ❌ Failure")
            
            total_episodes += 1
            
            # Save video
            # video_path = save_rollout_video(
            #     top_view_frames,
            #     wrist_view_frames,
            #     idx=total_episodes,
            #     success=success,
            #     task_description=task_desc,
            #     log_file=log_file
            # )
            
        # Task Summary
        task_rate = task_successes / cfg.num_trials_per_task
        print(f"   Task Success Rate: {task_rate:.2%}")
        log_file.write(f"   Task Success Rate: {task_rate:.2%}\n")
        log_file.flush()
        
        env.close()

    # Final Summary
    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    print("\n" + "=" * 80)
    print("📊 Evaluation Summary")
    print("=" * 80)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {overall_rate:.2%}")
    print(f"Logs saved to: {log_path}")
    print("=" * 80)
    
    log_file.write("\nFinal Summary\n")
    log_file.write(f"Total Episodes: {total_episodes}\n")
    log_file.write(f"Total Successes: {total_successes}\n")
    log_file.write(f"Overall Success Rate: {overall_rate:.2%}\n")
    log_file.close()



def main_all(cfg):
    print("=" * 80)
    print("🎯 LIBERO Simulation Evaluation")
    print("=" * 80)
    tasks = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]

    # Use a spawn context explicitly
    ctx = mp.get_context("spawn")

    results = {}
    # Import AFTER start method set (avoids some edge cases)
    # from suite_runner import run_suite

    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futures = {pool.submit(main, cfg, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                results[task] = fut.result()
            except Exception as e:
                print(f"[ERROR] Task '{task}' failed: {e}")

    print("All done. Results:", results)


if __name__ == "__main__":
    
    
# Set spawn before any CUDA/PyTorch code runs
    try:
        mp.set_start_method("spawn", force=True)
        # If you use torch.multiprocessing anywhere:
        import torch
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = tyro.cli(SimConfig)
    main_all(cfg)

