#!/bin/bash

#SBATCH --job-name=dker_hrnet-w48_8xb10_humanart-512x512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --constrain=h100
#SBATCH --output=dker_hrnet-w48_8xb10_humanart-512x512.out
#SBATCH --error=dker_hrnet-w48_8xb10_humanart-512x512.err

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