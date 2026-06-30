export PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/.venv/bin:$PATH
export PYTHON_PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=INFO

source .venv/bin/activate

echo "Using Python: $(which python)"
python --version
export WANDB_API_KEY=4d7d3d768aa4a5e26e287875fad303a3c4586fc6
export WANDB_MODE=online

export WANDB_PROJECT=VLA_layer_prunner
export WANDB_ENTITY=Robotics_VLA
export WANDB_MODE=online
export WANDB_ENABLE=true


DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/robocasa_100_demos_lerobot_5_chosen_tasks_v2"
MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
BATCH_SIZE=32
# MAX_STEPS=60000
MAX_STEPS=45000

# EXP_NAME=debug
EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_ROBOCASA_gr00tn15_CKA"
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"


cp examples/My_robocasa/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.My_robocasa.data_config:RobocasaDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size ${BATCH_SIZE} \
    --save-steps 20000 \
    --max-steps ${MAX_STEPS} \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to wandb