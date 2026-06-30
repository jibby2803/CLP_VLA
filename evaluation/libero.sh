#!/bin/bash
#SBATCH --job-name=evalgr15libero
#SBATCH --partition=main
#SBATCH --nodelist=worker-0,worker-1,worker-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --output=slurm_out/slurm-%j.out

export PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/.venv/bin:$PATH

source .venv/bin/activate

POLICY_PATH=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-21/02-17-23_bz32_maxstep100000_LIBERO_gr00tn15_COSINE_v1/checkpoint-100000
EXP_FOLDER=$(basename "$(dirname "$POLICY_PATH")")
CKPT_STEP=$(basename "$POLICY_PATH")
EXP_NAME="COSINE_${EXP_FOLDER}_${CKPT_STEP}"

# EXP_NAME=test

python evaluation/eval_libero.py \
  --args.model_type=gr00tn15 \
  --args.num_trials_per_task=50 \
  --args.exp_name=$EXP_NAME \
  --args.load_pruned_model=True \
  --args.pretrained_model_path=$POLICY_PATH