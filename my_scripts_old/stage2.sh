export PATH=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/miniconda3/envs/gr00t/bin:$PATH
export HF_HOME=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/hf_cache
export PYTHON_PATH=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=INFO 
conda activate gr00t

# Debug: show Python & CUDA info
echo "Using Python: $(which python)"
python --version
# nvidia-smi || echo "No GPU visible"

EXP_NAME="stage2_fully_finetune_from_stage1_v1"


# DATA_PATH="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/LIBERO/merged_libero_mask_noops_lerobot_10_v4" # scale 10%
# DATA_PATH="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/LIBERO/merged_libero_mask_noops_lerobot_40_v4" # scale 40%
DATA_PATH="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/LIBERO/merged_libero_mask_noops_lerobot_v4" # scale 100%

OUTPUT_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom/outputs"
CURRENT_DATE=$(date +%Y-%m-%d)

# MODEL_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/hf_cache/hub/models--nvidia--GR00T-N1-2B/snapshots/fc879581ca32f4f6d6e02cf0cc80452f6b0c3873"
# MODEL_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
MODEL_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom/outputs/2025-12-11/stage1_unfreeze_vlm_freeze_remaining_v1/checkpoint-20000"

# Run debug
# cp examples/My_libero/modality.json ${DATA_PATH}/meta/modality.json
# python scripts/gr00t_finetune.py \
#     --base-model-path ${MODEL_DIR} \
#     --dataset-path ${DATA_PATH} \
#     --data_config examples.My_libero.data_config:LiberoDataConfig \
#     --video-backend torchvision_av \
#     --tune-llm \
#     --tune-visual \
#     --tune-diffusion-model \
#     --tune-projector \
#     --batch-size 16 \
#     --save-steps 10000 \
#     --max-steps 60000 \
#     --num-gpus 1 \
#     --output-dir ${OUTPUT_DIR}/${CURRENT_DATE}/${EXP_NAME} \
#     --report-to tensorboard


cp examples/My_libero/modality.json ${DATA_PATH}/meta/modality.json
python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.My_libero.data_config:LiberoDataConfig \
    --video-backend torchvision_av \
    --tune-llm \
    --tune-visual \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size 24 \
    --save-steps 10000 \
    --max-steps 60000 \
    --num-gpus 4 \
    --output-dir ${OUTPUT_DIR}/${CURRENT_DATE}/${EXP_NAME} \
    --report-to tensorboard

# python scripts/gr00t_finetune.py \
#     --dataset-path /tmp/libero_spatial/ \
#     --data_config examples.Libero.custom_data_config:LiberoDataConfig \
#     --num-gpus 8 \
#     --batch-size 128 \
#     --output-dir /outputs/${EXP_NAME}/ \
#     --max-steps 60000