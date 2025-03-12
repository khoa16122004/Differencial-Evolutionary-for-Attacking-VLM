#!/bin/sh
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python main.py --image_dir /data/elo/khoatn/AttackVLM/images --target_dir /data/elo/khoatn/AttackVLM/target_image/samples --annotation_path /data/elo/khoatn/AttackVLM/annotations.txt --F 0.8 --CR 0.1 --sigma 0.05 --max_iter 100 --pop_size 80 --output_dir DE