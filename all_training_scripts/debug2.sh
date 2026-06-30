
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

export WANDB_PROJECT=VLA_layer_prunner
export WANDB_ENTITY=Robotics_VLA
export WANDB_MODE=online
export WANDB_ENABLE=true

# DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/gr16_distil/data/fractal20220817_data_lerobot"
# MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/debug/20260227_VR_H31_bodyshop_place_part2_eval_trym_stereo_rvt_speedup1"

MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/atan/test/20260313_gr00t15_best"
BATCH_SIZE=5
MAX_STEPS=60000

EXP_NAME="debug_atan"
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"

# EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_fractal_gr00tn15_base"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"

cp examples/atan/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.atan.data_config:VRH311LeftArmQuant2IncludeTaskProgressExcludeVelocityEffortConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size $BATCH_SIZE  \
    --save-steps 10000 \
    --max-steps $MAX_STEPS \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to wandb