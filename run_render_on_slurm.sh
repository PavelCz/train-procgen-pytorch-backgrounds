#!/bin/bash
# Wrapper script to run render_slurm_fix.py on SLURM
# This script clears SLURM/MPI environment before running Python

echo "Clearing SLURM/MPI environment variables..."

# Unset all SLURM-related MPI variables
unset SLURM_MPI_TYPE
unset SLURM_STEP_RESV_PORTS  
unset $(env | grep '^OMPI_' | cut -d= -f1)
unset $(env | grep '^PMI_' | cut -d= -f1)
unset $(env | grep '^MPI_' | cut -d= -f1)
unset $(env | grep '^I_MPI_' | cut -d= -f1)

# Set threading environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Environment cleaned. Running render_slurm_fix.py..."

# Run the Python script with all arguments passed through
python render_slurm_fix.py "$@"

