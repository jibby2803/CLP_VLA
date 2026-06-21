# export PATH=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/miniconda3/envs/gr00t/bin:$PATH
# export HF_HOME=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/hf_cache
# export PYTHON_PATH=/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/gr00t_custom:$PYTHONPATH
# export TORCH_DISTRIBUTED_DEBUG=INFO 

# path for vinr
export PATH=/home/binhng/conda_setup/miniconda3/envs/gr00t/bin:$PATH
export HF_HOME=/mnt/data/sftp/data/vla_intern/workspace/hf_home
export PYTHON_PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/gr00t_custom:$PYTHONPATH
export TORCH_DISTRIBUTED_DEBUG=INFO 


# source /projects/extern/kisski/kisski-spath/dir.project/VLA_3D/miniconda3/bin/activate
conda activate gr00t

# Debug: show Python & CUDA info
echo "Using Python: $(which python)"
python --version
# nvidia-smi || echo "No GPU visible"

# EXP_NAME="debug"
EXP_NAME="base_robocasa_100demos_v2_bz32_lr1e-4_24tasks_v1"

DATA_PATH="/mnt/data/sftp/data/vla_intern/workspace/data/ROBOCASA/robocasa_merged_24_tasks_100demos_v1"

OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/gr00t_custom/outputs"
CURRENT_DATE=$(date +%Y-%m-%d)

# MODEL_DIR="/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
MODEL_DIR="/mnt/data/sftp/data/vla_intern/workspace/model/GR00T-N1.5-3B"


cp examples/My_robocasa/modality.json ${DATA_PATH}/meta/modality.json
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --base-model-path ${MODEL_DIR} \
    --dataset-path ${DATA_PATH} \
    --data_config examples.My_robocasa.data_config:RobocasaDataConfig \
    --video-backend torchvision_av \
    --tune-diffusion-model \
    --tune-projector \
    --batch-size 32 \
    --save-steps 10000 \
    --max-steps 60000 \
    --num-gpus 1 \
    --output-dir ${OUTPUT_DIR}/${CURRENT_DATE}/${EXP_NAME} \
    --report-to tensorboard

