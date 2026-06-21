#!/bin/bash
#SBATCH -p kisski-h100,kisski
#SBATCH --job-name=gr00t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=150GB
#SBATCH -t 48:00:00
#SBATCH --exclude=ggpu102,ggpu103,ggpu104,ggpu105,ggpu106,ggpu107,ggpu108,ggpu109,ggpu110,ggpu111,ggpu112,ggpu113,ggpu114,ggpu115,ggpu116,ggpu117,ggpu118,ggpu119,ggpu120,ggpu121,ggpu122,ggpu123,ggpu124,ggpu125,ggpu126,ggpu127,ggpu128,ggpu129,ggpu130,ggpu131,ggpu132,ggpu133,ggpu134,ggpu135,ggpu136


python myutils/sleep.py