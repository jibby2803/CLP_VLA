
# path for vinr
export PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/.venv/bin:$PATH
# export HF_HOME=/mnt/data/sftp/data/vla_intern/workspace/hf_home
export PYTHON_PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=INFO 

# source /projects/extern/kisski/kisski-spath/dir.project/VLA_3D/miniconda3/bin/activate
source .venv/bin/activate

# Debug: show Python & CUDA info
echo "Using Python: $(which python)"
python --version
# nvidia-smi || echo "No GPU visible"


export WANDB_API_KEY=4d7d3d768aa4a5e26e287875fad303a3c4586fc6
export WANDB_MODE=online

export WANDB_PROJECT=VLA_layer_prunner
export WANDB_ENTITY=Robotics_VLA
export WANDB_MODE=online
export WANDB_ENABLE=true

DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/gr16_distil/data/fractal20220817_data_lerobot"
MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
BATCH_SIZE=64
MAX_STEPS=100000

# EXP_NAME="debug"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"
DATA_PATH_1="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/libero_10_no_noops_1.0.0_lerobot"
DATA_PATH_2="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/libero_object_no_noops_1.0.0_lerobot"
DATA_PATH_3="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/libero_goal_no_noops_1.0.0_lerobot"
DATA_PATH_4="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/libero_spatial_no_noops_1.0.0_lerobot"
DATA_PATH="$DATA_PATH_1 $DATA_PATH_2 $DATA_PATH_3 $DATA_PATH_4"
echo $DATA_PATH

EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_libero_gr00tn15_base"
# EXP_NAME=test
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"

cp examples/ipec_libero/modality.json ${DATA_PATH_1}/meta/modality.json
cp examples/ipec_libero/modality.json ${DATA_PATH_2}/meta/modality.json
cp examples/ipec_libero/modality.json ${DATA_PATH_3}/meta/modality.json
cp examples/ipec_libero/modality.json ${DATA_PATH_4}/meta/modality.json

CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.ipec_libero.data_config:MyDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size $BATCH_SIZE  \
    --save-steps 10000 \
    --max-steps $MAX_STEPS \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to wandb