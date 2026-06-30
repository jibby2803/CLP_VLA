
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

# export WANDB_PROJECT=GR00TN1.5 CORL2026
# export WANDB_ENTITY=trbaohoo #Robotics_VLA
export WANDB_PROJECT=CoRLRealworl
export WANDB_ENTITY=Robotics_VLA
export WANDB_MODE=online
export WANDB_ENABLE=true


DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/simplerenv_data/bridge_orig_lerobot" # banana drawer

MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
BATCH_SIZE=64
MAX_STEPS=200000

# EXP_NAME="debug"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"

EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_IPEC_BRIDGE_gr00tn15_BASE"
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"



cp examples/ipec_bridge/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune_base.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.ipec_bridge.data_config:BridgeDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size ${BATCH_SIZE} \
    --save-steps 50000 \
    --max-steps ${MAX_STEPS} \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to wandb