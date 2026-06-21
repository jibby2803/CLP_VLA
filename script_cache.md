# change dir
```bash
cd /projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom
```

## activate env
```bash
cd /projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom
source /projects/extern/kisski/kisski-spath/dir.project/VLA_3D/miniconda3/bin/activate
conda activate gr00t 
```

## install using uv
```bash
pip install uv
export UV_CACHE_DIR=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/uv_cache

uv pip install ...
uv pip install robosuite==1.4.0 bddl easydict tyro gym

rm -rf $UV_CACHE_DIR/*
```

## query debug
```
srun --pty -p kisski \
     --job-name=bdebug \
     --gpus=1 \
     --cpus-per-task=16 \
     --mem=80G \
     --time=48:00:00 \
     --exclude=ggpu177,ggpu179,ggpu184,ggpu193 \
     bash -i

srun --pty --job-name=gr00tdb \
     --partition=main \
     --nodes=1 \
     --ntasks=1 \
     --gpus=nvidia_h100_80gb_hbm3:4 \
     --cpus-per-task=32 \
     --mem=80G \
     --time=20:00:00 \
     bash -i

srun --pty --job-name=pi0 \
     --partition=main \
     --nodelist=worker-0,worker-1 \
     --nodes=1 \
     --ntasks=1 \
     --gpus=nvidia_h100_80gb_hbm3:3 \
     --cpus-per-task=48 \
     --mem=280G \
     --time=120:00:00 \
     bash -i

srun --pty --ntasks=1 --mem=12G --cpus-per-task=16 bash -i
 
```

## query to train
```
srun --pty -p kisski-h100,kisski \
     --job-name=bgr00t \
     --nodes=1 \
     --gpus=3 \
     --cpus-per-task=16 \
     --mem=200G \
     --time=48:00:00 \
     bash -i


srun --pty --job-name=gr15 \
--partition=main \
--nodelist=worker-0,worker-1 \
--nodes=1 \
--ntasks=1 \
--gpus=nvidia_h100_80gb_hbm3:1 \
--cpus-per-task=16 \
--mem=400G \
--time=148:00:00 \
bash -i

srun --pty --job-name=gr15 \
--partition=main \
--nodes=1 \
--ntasks=1 \
--gpus=nvidia_h100_80gb_hbm3:1 \
--cpus-per-task=16 \
--mem=600G \
--time=148:00:00 \
bash -i
```


## Finetune
```bash
bash my_scripts/libero_finetune.sh
bash my_scripts/libero_finetune_kisski2.sh
bash my_scripts/ft10.sh
bash my_scripts/ft40.sh
bash my_scripts/ft100.sh
bash my_scripts/debug.sh
bash my_scripts/debug.sh > debug.log 2>&1
bash my_scripts/ft_robocasa_30.sh
bash my_scripts/ft_robocasa_100.sh
```

