#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="synergetic"

# Partition for the job:
#SBATCH --partition gpu-a100

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-4:0:00

# Send yourself an email when the job:
#SBATCH --mail-user=culuwitikega@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
module load Anaconda3/2022.10
module load CUDA/12.2.0

export CONDA_ENVS_PATH=/data/projects/punim2453/conda_envs

eval "$(conda shell.bash hook)"
conda activate torch_env
# pip install openpyxl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH #run this terminal when you run python scripts in terminal
# The job command(s):
cd ./../src

python -u ensemble_train.py -e ensemble_100