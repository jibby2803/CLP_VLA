
# path for vinr
export PATH=/home/binhng/conda_setup/miniconda3/envs/gr15v2/bin:$PATH
# export HF_HOME=/mnt/data/sftp/data/vla_intern/workspace/hf_home
export PYTHON_PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=INFO 

# source /projects/extern/kisski/kisski-spath/dir.project/VLA_3D/miniconda3/bin/activate
conda activate gr15v2

# Debug: show Python & CUDA info
echo "Using Python: $(which python)"
python --version
# nvidia-smi || echo "No GPU visible"


export WANDB_API_KEY=4d7d3d768aa4a5e26e287875fad303a3c4586fc6
export WANDB_MODE=online

export WANDB_PROJECT=vrh3_bodyshop
export WANDB_ENTITY=Robotics_VLA
export WANDB_MODE=online
export WANDB_ENABLE=true

MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
BATCH_SIZE=24
MAX_STEPS=140000

# EXP_NAME="debug"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"
DATA_ROOT="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/bodyshop_placepart"

DATA_PATH_1=${DATA_ROOT}/place_part2_20260227_part2_trym
DATA_PATH_2=${DATA_ROOT}/place_part2_20260227_trym
DATA_PATH_3=${DATA_ROOT}/place_part2_20260302_trym
DATA_PATH_4=${DATA_ROOT}/place_part2_20260309_part1_trym
DATA_PATH_5=${DATA_ROOT}/place_part2_20260309_part2_trym
DATA_PATH_6=${DATA_ROOT}/place_part2_20260310_part2_trym
DATA_PATH_7=${DATA_ROOT}/place_part2_20260310_trym
DATA_PATH_8=${DATA_ROOT}/place_part2_20260312
DATA_PATH_9=${DATA_ROOT}/place_part2_correctip_20260313
DATA_PATH_10=${DATA_ROOT}/place_part2_correctip_20260314

DATA_PATH="$DATA_PATH_1 $DATA_PATH_2 $DATA_PATH_3 $DATA_PATH_4 $DATA_PATH_5 $DATA_PATH_6 $DATA_PATH_7 $DATA_PATH_8 $DATA_PATH_9 $DATA_PATH_10"
echo $DATA_PATH

EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_placepart_gr00tn15_base"
# EXP_NAME=test
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"

cp examples/alinh/modality.json ${DATA_PATH_1}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_2}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_3}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_4}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_5}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_6}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_7}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_8}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_9}/meta/modality.json
cp examples/alinh/modality.json ${DATA_PATH_10}/meta/modality.json


CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.alinh.data_config:VRH31OneHand3CamConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size $BATCH_SIZE  \
    --save-steps 20000 \
    --max-steps $MAX_STEPS \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to wandb