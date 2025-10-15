#!/bin/bash
#SBATCH --job-name=render_coinrun
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/render_%j.out
#SBATCH --error=logs/render_%j.err

# Clear MPI environment variables that can cause hangs
echo "Clearing MPI environment variables..."
unset SLURM_MPI_TYPE
for var in $(env | grep '^OMPI_\|^PMI_\|^MPI_\|^I_MPI_' | cut -d= -f1); do
    unset $var
    echo "  Unset $var"
done

# Set threading environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting render job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Set your model file path here
export COINRUN_BG_EXTRAHARD="/path/to/your/model.pth"

# Run the render script
python render_slurm_fix.py \
    --exp_name coinrun_hard_extrahard_bg_test \
    --env_name coinrun \
    --random_percent 50 \
    --distribution_mode hard \
    --param_name hard-500 \
    --start_level 300000 \
    --num_levels 1024 \
    --seed 1 \
    --model_file $COINRUN_BG_EXTRAHARD \
    --noview

echo "Render job completed with exit code: $?"

