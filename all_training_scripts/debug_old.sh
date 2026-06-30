
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

DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/data/debug/20260227_VR_H31_bodyshop_place_part2_eval_trym_stereo_rvt_speedup1"
# MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/cpkt/GR00T-N1.5-3B"
MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/vrh3-bodyshops-gripper-both-2026-03-14_11-13-49"
BATCH_SIZE=10
MAX_STEPS=100

EXP_NAME="debug"
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/${EXP_NAME}"

# EXP_NAME="bz${BATCH_SIZE}_maxstep${MAX_STEPS}_fractal_gr00tn15_base"
# OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXP_NAME}"

cp examples/abao/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.abao.data_config:VRH31GripperBoth \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size $BATCH_SIZE  \
    --save-steps 100 \
    --max-steps $MAX_STEPS \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR} \
    --report-to tensorboard