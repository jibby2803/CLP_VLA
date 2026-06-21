# Gr00t Custom

## Installation

```bash
# git clone <gr00t repo>
# cd <gr00t repo>
conda create -n gr00t python=3.10
conda activate gr00t
# pip install -e ".[base]"
# pip install --no-build-isolation flash-attn==2.7.1.post4

uv pip install -e ".[base]"
# uv pip install --no-build-isolation flash-attn==2.7.1.post4
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

uv pip install flash_attn-2.7.1.post4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

rm -rf $UV_CACHE_DIR/*

```

## Download pretrained checkpoint
```bash
export HF_HOME=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/hf_cache

huggingface-cli download nvidia/GR00T-N1-2B

huggingface-cli download nvidia/GR00T-N1.5-3B

```


## Finetune
- In the `examples` directory, create you folder
- Insides the created directory, for e.x `My_robocasa`: create `modality.json` and `data_config.py` which will defined the keys of data for model and data transformations
- In the `my_scripts`, create bash file to train. (Check `my_scripts/ft_robocasa_30.sh`), change model dir and data path to yours

NOTE: you can check `script_cache.md` to see some useful commands

```bash
bash my_scripts/libero_finetune.sh
bash my_scripts/libero_finetune_kisski2.sh

bash my_scripts/debug.sh
bash my_scripts/debug.sh > debug.log 2>&1
bash my_scripts/ft_robocasa_30.sh
bash my_scripts/ft_robocasa_100.sh
```
NOTE: If you want to train with multi gpus, you should change `dataloader_num_workers: int = 2` (small number)
## Eval
## ROBOCASA EVAL
Install robocasa simulation
```
pip install uv
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
uv pip install -e .

cd ..
git clone https://github.com/jibby2803/robocasa_regenerate.git ./robocasa
cd robocasa
uv pip install -e .

python robocasa/scripts/download_kitchen_assets.py  
python robocasa/scripts/setup_macros.py  

```

Change the robocasa path, model path, exp_name, etc. NOTE: args.gpus is just used for save log file (Sorry for my stupid !!!!). Check file: `scripts/eval_robocasa.py`.
Make sure that 25 trials use seed 0 and 25 trials use seed 1 for reproduce
```
python scripts/eval_robocasa.py \
   --args.exp-name base_robocasa_100demos_60k \
   --args.num-trials 25 \
  --args.seed 0 \
   --args.gpu 0

python scripts/eval_robocasa.py \
   --args.exp-name base_robocasa_100demos_60k \
   --args.num-trials 25 \
  --args.seed 1 \
   --args.gpu 1
```
