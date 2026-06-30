
import os 
import sys 

sys.path.insert(0, "/mnt/data/sftp/data/vla_intern/workspace/binh/2026/evaluation_setup/libero_benchmark")
sys.path.insert(0, "/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner")


import collections
import dataclasses
import logging
import math
import pathlib
import multiprocessing as mp  # added for parallel workers

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro
import torch
import time

from concurrent.futures import ProcessPoolExecutor, as_completed


# from myutils.pi0_infer import Pi0TorchInference, normalize_gripper_action, invert_gripper_action
# from evaluation.pi0_infer.pi0_inference import Pi0_inference
# from evaluation.smolvla_infer.smolvla_inference import SmolVLA_inference

from evaluation.gr00tn15_infer.gr00tn15_inference import Gr00tn15_inference


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def to_video_frame(arr):
    arr = np.asarray(arr)
    if any(s < 0 for s in arr.strides) or not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 3 and arr.shape[-1] > 4:
        arr = arr[..., :3]

    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        if arr.shape[-1] == 2:
            arr = arr[..., 0] 
        else:
            raise ValueError(f"Unexpected channel count: {arr.shape}")

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if 0.0 <= vmin and vmax <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    arr = np.ascontiguousarray(arr)
    return arr


@dataclasses.dataclass
class Args:
    pretrained_model_path: str = "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/VLA-Humanoid/outputs/train/2025-11-26/11-34-23_libero_100%_base_12layers_only/checkpoints/060000/pretrained_model"
    resize_size: int = 224
    infer_chunk: int = 10
    task_suite_name="libero_goal"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10 #50  # Number of rollouts per task

    seed: int = 7  # Random Seed (for reproducibility)
    exp_name: str = "test"
    model_type: str = "pi0"
    load_pruned_model: bool = True


def eval_libero(args: Args, task_suite_name:str=None) -> None:
    # Set random seed
    np.random.seed(args.seed)

    if task_suite_name is not None:
        args.task_suite_name = task_suite_name
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    log_dir = pathlib.Path(f"eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.task_suite_name}.log"
    
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    
    save_video_dir = pathlib.Path(f"eval_results/{args.exp_name}/videos/{args.task_suite_name}")
    save_video_dir.mkdir(parents=True, exist_ok=True)

    try:
        # mypolicy = Pi0TorchInference(args.pretrained_model_path, device=f"cuda:0")
        if args.model_type=="pi0":
            mypolicy = Pi0_inference(args.pretrained_model_path, args.infer_chunk)
            logging.info(f"Task {args.task_suite_name} | Successfully loaded {args.model_type} policy")
        elif args.model_type=="smolvla":
            mypolicy = SmolVLA_inference(args.pretrained_model_path, args.infer_chunk)
            logging.info(f"Task {args.task_suite_name} | Successfully {args.model_type} loaded policy")
        elif args.model_type=="gr00tn16":
            mypolicy = Gr00tn16_inference(args.pretrained_model_path, args.infer_chunk)
            logging.info(f"Task {args.task_suite_name} | Successfully {args.model_type} loaded policy")
        # elif args.model_type=="gr00tn15":
        #     mypolicy = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
        #     logging.info(f"Task {args.task_suite_name} | Successfully {args.model_type} loaded policy")
        elif args.model_type=="gr00tn15":
            mypolicy = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk, load_pruned_model=args.load_pruned_model)
            logging.info(f"Task {args.task_suite_name} | Successfully {args.model_type} loaded policy")
        else:
            print(f"{args.model_type} is not supported yet")
            pass
    except Exception as e:
        logging.info(f"Task {args.task_suite_name} | Failed to load policy: {e}")
        return
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        logging.info(f"Task_id: {task_id}")

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_images_wrist = []

            if task_episodes % 10 == 0:
                logging.info(f"Task_id: {task_id} | Starting episode {task_episodes+1}... | {task_description}")
            while t < max_steps + args.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                
                # start_time = time.time()
                action_chunk = mypolicy.get_libero_action(obs, task_description)
                # print(f"Inference speed: {time.time() - start_time} s")
                
                for act in action_chunk:
                    obs, reward, done, info = env.step(act.tolist())
                    t +=1
                    
                    replay_image = obs["agentview_image"][::-1, ::-1]
                    replay_images.append(to_video_frame(replay_image))
                    
                    replay_image_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    replay_images_wrist.append(to_video_frame(replay_image_wrist))
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                if done:
                    break
            
            task_episodes +=1
            total_episodes +=1
            suffix = "success" if done else "failure"
            
            imageio.mimwrite(
                pathlib.Path(f"{str(save_video_dir)}") / f"rollout_seed_{args.seed}_trial_{episode_idx}_wrist_{str(task_description)}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images_wrist],
                fps=10,
                codec="libx264",
            )
            imageio.mimwrite(
                pathlib.Path(f"{str(save_video_dir)}") / f"rollout_seed_{args.seed}_trial_{episode_idx}_static_{str(task_description)}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
                codec="libx264"
            )
  
        # Log current results
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    

def eval_libero_all(args:Args):
    print("=" * 80)
    print("🎯 LIBERO Simulation Evaluation")
    print("=" * 80)
    tasks = [
        'libero_10',
        'libero_spatid
    ]
    ctx = mp.get_context("spawn")
    results = dict()

    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futures = {pool.submit(eval_libero, args, task): task for task in tasks}
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

    tyro.cli(eval_libero_all)

'''
python evaluation/eval_libero.py \
    --args.exp_name=test \
    --args.pretrained_model_path=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/VLA_layer_prunner/output/2026-02-25/19-16-33_bz16_maxstep60000_libero_pi0_base/checkpoints/060000/pretrained_model

'''