# CLP: Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/abs/2606.20246)
[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-CLP--VLA-green)](https://clpvla.github.io/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](<huggingface_org_or_model_url>)
[![License](https://img.shields.io/badge/License-<license_name>-lightgrey.svg)](LICENSE)

<p align="center">
<img src="images/CLP.png" width="800" alt="CLP main figure">
</p>



<p align="center">
<em><one or two sentence summary of CLP: what CKA-Guided Layer Pruning does and the headline result, e.g. number of layers removed / speedup / accuracy retained></em>
</p>

## 📑 Table of Contents

- [TODO List](#todo-list)
- [📦 Installation](#-installation)
- [🔍 CKA-Guided Layer Pruning (CLP)](#-cka-guided-layer-pruning-clp)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## TODO List

- [ ] <Upload pruned / finetuned checkpoints to Hugging Face and update links below>
- [ ] <Add SimplerEnv evaluation instructions>


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

pip install --upgrade setuptools
pip install -e ".[base]"
pip install --no-build-isolation flash-attn==<flash_attn_version>
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


## 🔍 CKA-Guided Layer Pruning (CLP)

CLP identifies redundant layers in a pretrained VLA backbone using Centered Kernel Alignment (CKA) similarity between layer representations, then prunes them before finetuning on downstream manipulation tasks. The full workflow is described step-by-step below.

### Step 1 — Prepare your data

Convert your demonstrations into the **LeRobot v2.1** dataset format. E.g: https://huggingface.co/datasets/binhng/groceries_to_basket

### Step 2 — Identify redundant layers

Run a forward pass over the prepared dataset and cache the per-layer hidden states:

```bash
bash my_scripts/debug.sh
```

Then open the notebook below to visualize the layer-wise CKA similarity matrix and identify the high-CKA (redundant) layers to prune:

```bash
jupyter notebook notebook/cka.ipynb
```

### Step 3 — Finetune with CLP

Once the redundant layers are identified, finetune the pruned backbone:

```bash
bash my_scripts/finetune_cka.sh
```

<Add a short description of what this script does and the key flags it exposes, e.g. which layers to drop, batch size, max steps, output dir.>

### Step 4 — Deploy and evaluate

**LIBERO evaluation**

| Model | Description | 🤗 Download Link |
|-------|-------------|------------------|
| `<checkpoint_name>` | CLP-pruned, LIBERO | [Link](<hf_url>) |
| `<checkpoint_name>` | Base (no pruning), LIBERO | [Link](<hf_url>) |

```bash
bash evaluation/libero.sh
```

**SimplerEnv evaluation**

| Model | Description | 🤗 Download Link |
|-------|-------------|------------------|
| `<checkpoint_name>` | CLP-pruned, SimplerEnv | [Link](<hf_url>) |
| `<checkpoint_name>` | Base (no pruning), SimplerEnv | [Link](<hf_url>) |

<Add SimplerEnv setup + evaluation command here.>

**RoboCasa evaluation**

| Model | Description | 🤗 Download Link |
|-------|-------------|------------------|
| `<checkpoint_name>` | CLP-pruned, RoboCasa | [Link](<hf_url>) |
| `<checkpoint_name>` | Base (no pruning), RoboCasa | [Link](<hf_url>) |

```bash
bash evaluation/robocasa0.sh
bash evaluation/robocasa1.sh
```




## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{nguyen2026finetuningvisionlanguageactionmodelsrequires,
      title={Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think}, 
      author={Gia-Binh Nguyen and Trong-Bao Ho and Thien-Loc Ha and Khoa Vo and Philip Lund Møller and Quang T. Nguyen and Long Dinh and Tuan Dam and Vu Duong and Tung M. Luu and Trung Le and Tran Nguyen Le and Minh Vu and An Thai Le and Ngan Le and Daniel Sonntag and James Zou and Jan Peters and Duy M. H. Nguyen and Ngo Anh Vien},
      year={2026},
      url={https://arxiv.org/abs/2606.20246}, 
}
```


## Acknowledgement

This repository builds on [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) and [lerobot](https://github.com/huggingface/lerobot), with evaluation environments from [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), and [RoboCasa](https://github.com/robocasa/robocasa). Thanks for their excellent open-source work!