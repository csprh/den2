#!/bin/bash

#SBATCH --job-name=cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M

module load CUDA
module load apps/matlab/2018a

cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR
cd /mnt/storage/home/csprh/code/HUAWEI/DNCNN/TrainingCodes/DnCNNHuawei
#matlab -nodisplay -nosplash -r getDataOuter > outfile.txt < /dev/null
matlab -nodisplay -nosplash -r Demo_Train_model__Huwei_All
