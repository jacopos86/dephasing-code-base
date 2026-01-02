#!/bin/bash
# -----------------------------
# NERSC HPC environment setup
# -----------------------------

# Restore system defaults first
source /opt/cray/pe/cpe/25.03/restore_lmod_system_defaults.sh

# Load required modules
module load gcc-native/13.2
module load cray-mpich

# -----------------------------
# Build mode and GPU flag
# -----------------------------
export BUILD_MODE=delta
export INSTALL_PYCUDA=0
export PETSC_DIR=/pscratch/sd/j/jsimoni/PYDEPHASING/petsc
export PETSC_ARCH=arch-linux-c-opt

# -----------------------------
# Verify MPI
# -----------------------------
echo "MPI compiler:"
which mpicc
mpicc -show
# -----------------------------
# Finished
# -----------------------------
echo "DELTA environment setup complete."
