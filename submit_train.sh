#!/bin/bash
#SBATCH --account=gfdlport
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --job-name=SPEAR_emulator_training
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uriel.ramirez@noaa.gov

module load rdhpcs-conda/25.11.0
conda activate /scratch4/GFDL/gfdlscr/Uriel.Ramirez/conda/envs/WUT

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

cd /scratch4/GFDL/gfdlscr/Uriel.Ramirez/DEV/LIGHTING
echo $LD_LIBRARY_PATH

echo "Training the model ... "
echo "---------------------------------------------"

python train_driver.py
