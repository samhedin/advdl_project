#!/usr/bin/env bash
#SBATCH -n 16
#SBATCH --gpus 1
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --error /proj/attention-cpu/users/%u/projects/advdl_project/logs/%J_slurm_2b.err
#SBATCH --output /proj/attention-cpu/users/%u/projects/advdl_project/logs/%J_slurm_2b.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Experiment 2b"
nvidia-smi
module load Anaconda/2021.05-nsc1
pushd /proj/attention-cpu/users/x_cuoda/projects/advdl_project/src/pixelcnn
pwd
conda activate /proj/attention-cpu/users/x_cuoda/.conda/advdl

python main.py -x 100 -l 0.0002 --exp_name "2b"