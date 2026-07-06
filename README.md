<p align="center">
  <h1 align="center">CLP: Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think</h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.20246"><img src="https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg" alt="ArXiv"></a>
  <a href="https://clpvla.github.io/"><img src="https://img.shields.io/badge/🌐%20Project%20Page-CLP--VLA-green" alt="Project Page"></a>
  <a href="<huggingface_org_or_model_url>"><img src="https://img.shields.io/badge/🤗%20Model-HuggingFace-blue" alt="Hugging Face"></a>
</p>


<p align="center">
<img src="images/CLP.png" width="800" alt="CLP main figure">
</p>

<p align="center">
<em><one or two sentence summary of CLP: what CKA-Guided Layer Pruning does and the headline result, e.g. number of layers removed / speedup / accuracy retained></em>
</p>

## 📑 Table of Contents

- [TODO List](#todo-list)
- [📦 Installation](#-installation)
- [🚀 Part 1: Quickstart — Finetune & Evaluate CLP on Your Own Dataset](#-part-1-quickstart--finetune--evaluate-clp-on-your-own-dataset)
- [🔁 Part 2: Reproducing Paper Experiments](#-part-2-reproducing-paper-experiments)
  - [1. LIBERO](#1-libero)
    - [1.1 LIBERO Evaluation](#11-libero-evaluation)
    - [1.2 LIBERO Finetuning](#12-libero-finetuning)
  - [2. ROBOCASA](#2-robocasa)
    - [2.1 ROBOCASA Evaluation](#21-robocasa-evaluation)
    - [2.2 ROBOCASA Finetuning](#22-robocasa-finetuning)
  - [3. SimplerEnv](#3-simplerenv)
    - [3.1 SimplerEnv Setup](#31-simplerenv-setup)
    - [3.2 Checkpoints](#32-checkpoints)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## TODO List

- [ ] <Upload remaining pruned / finetuned checkpoints to Hugging Face and update links below>
- [ ] <Add CLP server/client evaluation scripts for SimplerEnv (youliangtan/SimplerEnv fork), wiring `Gr00tPolicy` into Google Robot / WidowX+Bridge tasks>


## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/jibby2803/CLP_VLA.git
cd CLP_VLA
```

Create an environment and install the base dependencies (this repo builds on the [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) codebase):

```bash
conda create -n clp_vla python=3.10
conda activate clp_vla

pip install uv
uv pip install --upgrade setuptools
uv pip install -e ".[base]"

# install flash attention
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

uv pip install flash_attn-2.7.1.post4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

To run LIBERO / RoboCasa simulation for evaluation:

```bash
pip install uv
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite && uv pip install -e . && cd ..

git clone <robocasa_repo_url> ./robocasa
cd robocasa && uv pip install -e . && cd ..
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py
```

---

## 🚀 Part 1: Quickstart — Finetune & Evaluate CLP on Your Own Dataset

This section is a **detailed, step-by-step guide for running one specific model on one specific dataset** — i.e. how to bring your own robot demonstrations, apply CKA-Guided Layer Pruning (CLP), finetune, and evaluate. If you instead want to reproduce the exact experiments from the paper (LIBERO, RoboCasa, SimplerEnv, with all model variants), skip ahead to [Part 2](#-part-2-reproducing-paper-experiments).

### Step 1 — Prepare your data

Convert your demonstrations into the **LeRobot v2.1** dataset format. E.g: https://huggingface.co/datasets/binhng/groceries_to_basket

### Step 2 — Identify redundant layers (CKA analysis)

Run a forward pass over your prepared dataset and cache the per-layer hidden states. This is a debug/analysis-only run — it loads the full pretrained backbone, runs a single training step, and dumps each transformer layer's hidden states to disk (`save_hidden_state=True`, capped at 1 step by default):

```bash
bash my_scripts/debug.sh
```

Key flags in `my_scripts/debug.sh` you may want to edit:
- `MODEL_DIR` — path to the base pretrained checkpoint (e.g. GR00T-N1.5-3B).
- `DATA_PATH` — path to your LeRobot-format dataset from Step 1.
- `--save-hidden-state True` — enables the hidden-state dump (off by default everywhere else).
- `--save-hidden-state-dir` / `--save-hidden-state-max-steps` — where to write the `.npy` files and how many steps to capture (default: 1 step, then auto-disables).

Then open the notebook below to visualize the layer-wise CKA similarity matrix and identify the high-CKA (redundant) layers to prune:

```bash
jupyter notebook notebook/cka.ipynb
```

The notebook outputs a list of layer indices to **keep** for both the backbone (VLM) and the action head (DiT / VL self-attention). Update `kept_layer_idx_list` in `gr00t/model/backbone/eagle_backbone.py` and `kept_layer_idx_list_dit` / `kept_layer_idx_list_vl_self_attn` in `gr00t/model/action_head/flow_matching_action_head.py` with the layers you've identified.

### Step 3 — Finetune with CLP

Once the redundant layers are identified, finetune the pruned backbone on your dataset:

```bash
bash my_scripts/train_libero.sh
```

(or copy this script and point it at your own `DATA_PATH` / `data_config` — see `my_scripts/train_libero.sh` and `my_scripts/AKHOA_closepot_cka.sh` for examples on different datasets).

Key flags exposed by `scripts/gr00t_finetune.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model-path` | — | Path to the **full** pretrained checkpoint. |
| `--dataset-path` | — | Path to your LeRobot-format dataset. |
| `--data_config` | — | `module:ClassName` for your dataset's modality config / transforms. |
| `--prune-model` | `True` | Whether to prune the architecture (using the `kept_layer_idx_list*` you set in Step 2) after loading the full pretrained weights. Set to `False` to train a non-pruned baseline. |
| `--tune-llm` / `--tune-visual` | `False` | Whether to finetune the backbone's LLM / vision tower. |
| `--tune-projector` / `--tune-diffusion-model` | `True` | Whether to finetune the action head's projector / DiT. |
| `--batch-size`, `--max-steps`, `--save-steps` | — | Standard training hyperparameters. |
| `--save-hidden-state` | `False` | Debug-only; leave off for real training runs (see Step 2). |
| `--output-dir` | `/tmp/gr00t` | Where checkpoints get saved. |

> **Why pruning order matters:** training always loads the **full** pretrained checkpoint first (`pruned=False`), then prunes the in-memory architecture afterward via `--prune-model`. This avoids any shape/key mismatch that would happen if you tried to load full pretrained weights directly into an already-pruned architecture.

### Step 4 — Deploy and evaluate

Evaluate your finetuned, pruned checkpoint in LIBERO:

```bash
bash evaluation/libero.sh
```

Key flags in `evaluation/eval_libero.py` (set via `evaluation/libero.sh`):

| Flag | Default | Description |
|------|---------|-------------|
| `--args.pretrained_model_path` | — | Path to your finetuned checkpoint directory. |
| `--args.load_pruned_model` | `True` | Build the **pruned** architecture before loading weights — use this for a finetuned checkpoint saved after pruning (Step 3). Set to `False` only when evaluating a base/unpruned checkpoint. |
| `--args.model_type` | `pi0` | Set to `gr00tn15` for this codebase. |
| `--args.task_suite_name` | `libero_goal` | Which LIBERO task suite to run. |
| `--args.num_trials_per_task` | `10` | Number of rollouts per task. |

> **Why this flag matters:** evaluation builds the architecture **pruned first, then** loads your finetuned weights — the opposite order from training, but necessary because the checkpoint you're loading is *already pruned*. `--args.load_pruned_model True` ensures the architecture matches the saved `state_dict` exactly.

For RoboCasa or your own custom simulation/real-robot eval harness, follow the same pattern: build a `Gr00tPolicy(..., load_pruned_model=<True/False to match your checkpoint>)` and call `policy.get_action(obs)`.

---

## 🔁 Part 2: Reproducing Paper Experiments

This section provides scripts to **reproduce the simulation experiments reported in the paper**, organized by benchmark. Each benchmark section is split into **Evaluation** (run our released checkpoints) and **Finetuning** (train the CLP model yourself from scratch).

### 1. LIBERO

<details>
<summary><b>Click to expand: Evaluation & Finetuning</b></summary>

#### 1.1 LIBERO Evaluation

We release the following checkpoints reported in our paper:

| Model | Description | 🤗 Download Link |
|-------|-------------|------------------|
| `<checkpoint_name>` | Base (no pruning), LIBERO | [Coming soon](<hf_url>) |
| `<checkpoint_name>` | CLP-pruned (CKA), LIBERO | [Coming soon](<hf_url>) |

**Step 1 — Set up the LIBERO simulator.**

```bash
pip install uv
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite && uv pip install -e . && cd ..
```

(See [Installation](#-installation) above for the full LIBERO/RoboCasa simulator setup.)

**Step 2 — Run the evaluation.**

```bash
bash evaluation/libero.sh
```

which under the hood runs:

```bash
python evaluation/eval_libero.py \
    --args.model_type=gr00tn15 \
    --args.pretrained_model_path="<path_to_checkpoint>" \
    --args.load_pruned_model=True \
    --args.task_suite_name="libero_goal" \
    --args.num_trials_per_task=50 \
    --args.exp_name="<exp_name>"
```

Key evaluation arguments:

| Argument | Description |
|----------|-------------|
| `--args.pretrained_model_path` | Path to a checkpoint dir (contains `model.safetensors` / `*.safetensors` shards and `config.json`). |
| `--args.load_pruned_model` | `True` to build the **pruned** architecture before loading weights (use for the CLP checkpoint above). Set to `False` when evaluating the **Base (no pruning)** checkpoint. |
| `--args.task_suite_name` | LIBERO suite to evaluate (e.g. `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`). |
| `--args.num_trials_per_task` | Number of rollout episodes per task. |
| `--args.exp_name` | Name used for the results/video/log output directory, written to `eval_results/<exp_name>/`. |

#### 1.2 LIBERO Finetuning

| Dataset | Description | 🤗 Download Link |
|---------|-------------|------------------|
| `merged_libero_mask_noops_lerobot_v4` | LIBERO (LeRobot v2.1 format) | [binhng/merged_libero_mask_noops_lerobot_v4](https://huggingface.co/datasets/binhng/merged_libero_mask_noops_lerobot_v4) |

**Step 1 — Download the base GR00T-N1.5 pretrained checkpoint** and the LIBERO dataset from the table above (or your own, converted to LeRobot v2.1 format — see [Part 1, Step 1](#step-1--prepare-your-data)).

**Step 2 — Identify the redundant layers to prune**, following [Part 1, Step 2](#step-2--identify-redundant-layers-cka-analysis) (`bash my_scripts/debug.sh` then `notebook/cka.ipynb`), and set `kept_layer_idx_list*` accordingly in `gr00t/model/backbone/eagle_backbone.py` / `gr00t/model/action_head/flow_matching_action_head.py`.

**Step 3 — Launch training.**

```bash
bash my_scripts/my_libero2.sh
```

which under the hood runs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path <path_to_GR00T-N1.5_checkpoint> \
    --dataset-path <path_to_libero_dataset> \
    --data_config examples.My_libero.data_config:LiberoDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --prune-model True \
    --batch-size 32 \
    --save-steps 10000 \
    --max-steps 100000 \
    --num-gpus 1 \
    --output-dir <output_dir> \
    --report-to wandb
```

Key training arguments:

| Argument | Description |
|----------|-------------|
| `--base-model-path` | Path to the **full** pretrained GR00T-N1.5 checkpoint. |
| `--dataset-path` | Path to the LIBERO dataset from Step 1. |
| `--prune-model` | `True` to prune the architecture (using the `kept_layer_idx_list*` from Step 2) after loading the full pretrained weights — this is what makes the run "CLP" rather than "Base". |
| `--tune-projector` / `--tune-diffusion-model` | Finetune the action head's projector / DiT. |
| `--batch-size`, `--max-steps`, `--save-steps` | Standard training hyperparameters. |
| `--num-gpus` | Set `>1` to automatically launch via `torchrun`. |
| `--output-dir` | Where checkpoints and logs are written. |
| `--report-to` | `wandb` to log to Weights & Biases, `tensorboard` for local logging. |

</details>


### 2. ROBOCASA

<details>
<summary><b>Click to expand: Evaluation & Finetuning</b></summary>

#### 2.1 ROBOCASA Evaluation

We release the following checkpoints reported in our paper:

| Model | Description | 🤗 Download Link |
|-------|-------------|------------------|
| `<checkpoint_name>` | Base (no pruning), RoboCasa | [Coming soon](<hf_url>) |
| `<checkpoint_name>` | CLP-pruned (CKA), RoboCasa | [Coming soon](<hf_url>) |

**Step 1 — Install the RoboCasa simulator.**

```bash
pip install uv
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite && uv pip install -e . && cd ..

git clone <robocasa_repo_url> ./robocasa
cd robocasa && uv pip install -e . && cd ..
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py
```

**Step 2 — Run the evaluation.**

```bash
bash evaluation/robocasa0.sh
bash evaluation/robocasa1.sh
```

Point each script at your checkpoint and set `--args.load_pruned_model=True` for the CLP checkpoint, or `False` for the Base (no pruning) checkpoint — same convention as LIBERO evaluation above.

Key evaluation arguments (same flags as [LIBERO evaluation](#11-libero-evaluation), plus):

| Argument | Description |
|----------|-------------|
| `--args.pretrained_model_path` | Path to a checkpoint dir. |
| `--args.load_pruned_model` | `True` for the CLP checkpoint, `False` for Base. |
| `--args.num_trials_per_task` | Number of rollout episodes per task. |
| `--args.exp_name` | Results/log output directory name. |

#### 2.2 ROBOCASA Finetuning

| Dataset | Description | 🤗 Download Link |
|---------|-------------|------------------|
| `robocasa_30_demos_lerobot_5_chosen_tasks_v3` | RoboCasa, 5 chosen tasks, 30 demos each (LeRobot v2.1 format) | [binhng/robocasa_30_demos_lerobot_5_chosen_tasks_v3](https://huggingface.co/datasets/binhng/robocasa_30_demos_lerobot_5_chosen_tasks_v3) |

The procedure is identical to [LIBERO finetuning](#12-libero-finetuning) — the same Steps 1-2 apply (download the base GR00T-N1.5 checkpoint, identify redundant layers via CKA). The only difference is the dataset and data config:

```bash
bash my_scripts/robocasa_cka.sh
```

which under the hood runs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path <path_to_GR00T-N1.5_checkpoint> \
    --dataset-path <path_to_robocasa_dataset> \
    --data_config examples.My_robocasa.data_config:RobocasaDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --prune-model True \
    --batch-size 32 \
    --save-steps 10000 \
    --max-steps 100000 \
    --num-gpus 1 \
    --output-dir <output_dir> \
    --report-to wandb
```

</details>


### 3. SimplerEnv

<details>
<summary><b>Click to expand: Evaluation</b></summary>

#### 3.1 SimplerEnv Setup

We follow [youliangtan/SimplerEnv](https://github.com/youliangtan/SimplerEnv) (a GR00T-compatible fork of [SimplerEnv](https://github.com/simpler-env/SimplerEnv)) to set up the simulation environment.

**Step 1 — Install system dependencies.**

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
```

**Step 2 — Clone and install the SimplerEnv fork.**

```bash
git clone https://github.com/youliangtan/SimplerEnv.git
cd SimplerEnv
# follow the installation instructions in this repo's README to set up
# the SAPIEN/ManiSkill2 simulator, Google Robot, and WidowX+Bridge environments
```

> 🚧 The CLP-specific server/client evaluation scripts that wire `Gr00tPolicy` (from [`gr00t/model/policy.py`](gr00t/model/policy.py)) into this SimplerEnv fork are not yet included in this repo — see the [TODO list](#todo-list). Once added, evaluation will follow the same pattern as LIBERO/RoboCasa above: load a checkpoint with `--load_pruned_model=True` for the CLP model or `False` for the Base model, then roll out episodes against the Google Robot / WidowX+Bridge tasks provided by the SimplerEnv fork.

**Dataset.** Bridge (WidowX) data used for SimplerEnv finetuning, in LeRobot format:

| Dataset | Description | 🤗 Download Link |
|---------|-------------|------------------|
| `bridge_orig_lerobot` | Bridge (WidowX), original demos (LeRobot format) | [IPEC-COMMUNITY/bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) |

#### 3.2 Checkpoints

| Model | Description | 🤗 Download Link |
|-------|-------------|------------------|
| `gr15_base_bridge_v1` | Base (no pruning), Bridge / SimplerEnv | [binhng/gr15_base_bridge_v1](https://huggingface.co/binhng/gr15_base_bridge_v1) |
| `gr15_cka_bridge_v1` | CLP-pruned (CKA), Bridge / SimplerEnv | [binhng/gr15_cka_bridge_v1](https://huggingface.co/binhng/gr15_cka_bridge_v1) |

</details>

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{nguyen2026finetuning,
  title={Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think},
  author={Nguyen, Gia-Binh and Ho, Trong-Bao and Ha, Thien-Loc and Vo, Khoa and M{\o}ller, Philip Lund and Nguyen, Quang T and Dinh, Long and Dam, Tuan and Duong, Vu and Luu, Tung M and others},
  journal={arXiv preprint arXiv:2606.20246},
  year={2026}
}
```


## Acknowledgement

This repository builds on [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) and [lerobot](https://github.com/huggingface/lerobot), with evaluation environments from [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), and [RoboCasa](https://github.com/robocasa/robocasa). Thanks for their excellent open-source work!