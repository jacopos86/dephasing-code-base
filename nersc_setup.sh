#!/bin/bash
# -----------------------------
# NERSC HPC environment setup
# -----------------------------

module purge
# Restore system defaults first
source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh

# Load required modules
module load python
module load PrgEnv-gnu

# -----------------------------
# Build mode and GPU flag
# -----------------------------
export BUILD_MODE=nersc
export INSTALL_PYCUDA=0
export DOWNLOAD_EXAMPLES=1
export DOWNLOAD_TESTS3=0
export BUILD_TESTS="1 2"

# -----------------------------
# Verify MPI
# -----------------------------
echo "MPI compiler:"
which mpicc
mpicc -show

# -----------------------------
# Configure Python environment
# -----------------------------
make configure

# -----------------------------
#  Build / install
# -----------------------------

make build

make install

# -----------------------------
#   TESTS
# -----------------------------

make test

# -----------------------------
# Finished
# -----------------------------
echo "NERSC environment setup complete."
