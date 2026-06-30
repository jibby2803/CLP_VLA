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

# NOTE: set WANDB_API_KEY in your shell env / secrets manager before running this
# script -- do NOT hardcode it here. e.g.:
#   export WANDB_API_KEY=<your_key>
# export WANDB_MODE=online

export WANDB_PROJECT=CoRLRealworl
export WANDB_ENTITY=Robotics_VLA
export WANDB_MODE=online
export WANDB_ENABLE=true


DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/ALOHA_BIMANUAL/fold_short"

# /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/ALOHA_BIMANUAL/fly_towel
# /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/ALOHA_BIMANUAL/fold_short

MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
BATCH_SIZE=32
MAX_STEPS=45000

# Whether to prune the architecture after loading the full pretrained weights.
# Training always loads the FULL pretrained checkpoint first (avoids shape/key
# mismatches), then prunes afterward if PRUNE_MODEL=True.
PRUNE_MODEL=True

# EXP_NAME="debug"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"

EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_ALOHA_foldshort_gr00tn15_CKA_v1"
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"



cp examples/corl_aloha_2_arms/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.corl_aloha_2_arms.data_config:AlohaTwoArmConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --prune-model ${PRUNE_MODEL} \
    --batch-size ${BATCH_SIZE} \
    --save-steps 5000 \
    --max-steps ${MAX_STEPS} \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to wandb