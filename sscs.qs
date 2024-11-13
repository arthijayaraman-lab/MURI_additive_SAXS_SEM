#!/bin/bash -l

#SBATCH --partition=gpu-t4
#SBATCH --job-name="sscs"
#SBATCH --time=0-6:00:00
#SBATCH -o stdout_%j.txt
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --export=NONE
#SBATCH --mail-user='aanishp@udel.edu'
#SBATCH --mail-type=ALL

vpkg_require anaconda/5.3.1:python3

cd  /home/3352/

source activate gpu

cd  /lustre/jayaraman_lab/users/3352/MURI_additive_SAXS_SEM/

python3 scattering_code_iq1_parallel.py