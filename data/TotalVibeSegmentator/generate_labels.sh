#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --partition=low-prio-gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=1                                       # Node count
#SBATCH --cpus-per-task=12                              # Number of CPU cores per task
#SBATCH --mem=45gb                                      # Job memory request
#SBATCH --array=0-56                                    # Launches jobs indices from 0 to 56
#SBATCH --output=output_%J_%i.log                       # %J is array job ID, %i is array index
#SBATCH --gres=gpu:1                                    # Requesting GPUs

srun python run_TotalVibeSegmentator.py --data_csv ./data/mri_dxa_ident_lastname_small_file_$SLURM_ARRAY_TASK_ID.csv --img ./spinenet-ukbb-stitching/UKBB_stitched/  --out_path ./TotalVibeSegmentator/Output/