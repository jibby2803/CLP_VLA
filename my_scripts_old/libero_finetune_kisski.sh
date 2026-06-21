export PATH=/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/miniconda3/envs/gr00t/bin:$PATH
export HF_HOME=$PROJECT_DIR/VLA/hf_cache
export PYTHON_PATH=/mnt/vast-kisski/projects/kisski-umg-fairpact-2/VLA/binh/gr00t_custom:$PYTHONPATH
conda activate gr00t

# Debug: show Python & CUDA info
echo "Using Python: $(which python)"
python --version
# nvidia-smi || echo "No GPU visible"

EXP_NAME="test1"
# DATA_PATH="/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/LIBERO/merged_libero_scale_10_mask_depth_noops_lerobot"
DATA_PATH="/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/cache/hf_cache/lerobot/binhng/merged_libero_mask_depth_noops_lerobot_10"
OUTPUT_DIR="/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/gr00t_custom/outputs"

# Run debug
# cp examples/My_libero/modality.json ${DATA_PATH}/meta/modality.json
# python scripts/gr00t_finetune.py \
#     --dataset-path ${DATA_PATH} \
#     --data_config examples.My_libero.data_config:LiberoDataConfig \
#     --video-backend torchvision_av \
#     --tune-llm \
#     --tune-visual \
#     --tune-diffusion-model \
#     --tune-projector \
#     --batch-size 1 \
#     --save-steps 10000 \
#     --max-steps 60000 \
#     --num-gpus 1 \
#     --output-dir ${OUTPUT_DIR}/${EXP_NAME} \
#     --report-to tensorboard

cp examples/My_libero/modality.json ${DATA_PATH}/meta/modality.json
python scripts/gr00t_finetune.py \
    --dataset-path ${DATA_PATH} \
    --data_config examples.My_libero.data_config:LiberoDataConfig \
    --video-backend torchvision_av \
    --tune-llm \
    --tune-visual \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size 32 \
    --save-steps 10000 \
    --max-steps 60000 \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR}/${EXP_NAME} \
    --report-to tensorboard

# python scripts/gr00t_finetune.py \
#     --dataset-path /tmp/libero_spatial/ \
#     --data_config examples.Libero.custom_data_config:LiberoDataConfig \
#     --num-gpus 8 \
#     --batch-size 128 \
#     --output-dir /outputs/${EXP_NAME}/ \
#     --max-steps 60000