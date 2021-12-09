#!/usr/bin/env bash
#SBATCH -n 16
#SBATCH --gpus 1
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --error /proj/attention-cpu/users/%u/projects/advdl_project/logs/%J_slurm_ssd1000.err
#SBATCH --output /proj/attention-cpu/users/%u/projects/advdl_project/logs/%J_slurm_ssd1000.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Experiment SSD1000"
nvidia-smi
module load Anaconda/2021.05-nsc1
pushd /proj/attention-cpu/users/x_cuoda/projects/advdl_project/src/pixelcnn
pwd
conda activate /proj/attention-cpu/users/x_cuoda/.conda/advdl

python main.py -x 1000 -l 0.0002 --exp_name "ssd1000" \
    --smooth 1 --resume_from 10 \
    -r models/ssd1000/pcnn_lr:0.00020_nr-resnet5_nr-filters160_noise-03_9.pth