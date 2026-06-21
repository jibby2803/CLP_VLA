source .venv/bin/activate

# python eval/eval_robocasa.py \
#     --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-13/16-38-09_bz32_maxstep60000_ROBOCASA_gr00tn15_BASE \
#     --args.exp-name gr00t_base_robocasa_100demos_60k_5tasks \
#     --args.num-trials 25 \
#     --args.seed 1 \
#     --args.gpu 1

# python eval/eval_robocasa.py \
#     --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-13/16-40-30_bz32_maxstep60000_ROBOCASA_gr00tn15_keep34 \
#     --args.exp-name gr00t_KEEP34_robocasa_100demos_60k_5tasks \
#     --args.num-trials 25 \
#     --args.seed 1 \
#     --args.gpu 1

# python eval/eval_robocasa.py \
#     --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-14/21-50-42_bz32_maxstep60000_ROBOCASA_gr00tn15_CKA \
#     --args.exp-name gr00t_CKA_robocasa_100demos_60k_5tasks \
#     --args.num-trials 25 \
#     --args.seed 1 \
#     --args.gpu 1

# python eval/eval_robocasa.py \
#     --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-13/16-44-00_bz32_maxstep60000_ROBOCASA_gr00tn15_keep24 \
#     --args.exp-name gr00t_keep24_robocasa_100demos_60k_5tasks \
#     --args.num-trials 25 \
#     --args.seed 1 \
#     --args.gpu 1

# python eval/eval_robocasa.py \
#     --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-13/16-40-30_bz32_maxstep60000_ROBOCASA_gr00tn15_keep34 \
#     --args.exp-name gr00t_KEEP34_robocasa_100demos_60k_5tasks_2nd \
#     --args.num-trials 25 \
#     --args.seed 1 \
#     --args.gpu 1



# python eval/eval_robocasa.py \
#     --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-15/17-30-58_bz32_maxstep60000_ROBOCASA_gr00tn15_CKA_AND_COSINE \
#     --args.exp-name gr00t_COSINE_CKA_robocasa_100demos_60k_5tasks_2nd \
#     --args.num-trials 25 \
#     --args.seed 1 \
#     --args.gpu 1

python eval/eval_robocasa.py \
    --args.model_dir /mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/Gr00tN1.5_prunner/outputs/2026-05-13/16-47-03_bz32_maxstep60000_ROBOCASA_gr00tn15_keep14 \
    --args.exp-name keep14 \
    --args.num-trials 25 \
    --args.seed 1 \
    --args.gpu 1