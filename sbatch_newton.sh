#!/bin/bash

#SBATCH --job-name=reward_controlnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --constrain=h100
#SBATCH --output=ft_ade20k.out
#SBATCH --error=ft_ade20k.err

module load anaconda
module load cuda/cuda-12.1
module load gcc/gcc-11.2.0
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate reward

echo "NVCC version:"
nvcc --version
echo
echo "GCC version:"
gcc -v 2>&1


bash tools/dist_train.sh configs/body_2d_keypoint/dekr/coco/dker_hrnet-w48_8xb10_humanart-512x512.py 2 --work-dir work_dirs/dekr_hrnet-w48_2xb20-140e_humanart-512x512 --auto-scale-lr --amp