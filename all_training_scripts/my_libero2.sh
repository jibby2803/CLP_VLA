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


DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/my_libero_v30/my_libero_v21_old"
MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
BATCH_SIZE=32
MAX_STEPS=80000

# EXP_NAME="debug"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"

EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_LIBERO_gr00tn15_cka_v1"
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"



cp examples/My_libero/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.My_libero.data_config:LiberoDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size ${BATCH_SIZE} \
    --save-steps 10000 \
    --max-steps ${MAX_STEPS} \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR}/${CURRENT_DATE}/${EXP_NAME} \
    --report-to wandb