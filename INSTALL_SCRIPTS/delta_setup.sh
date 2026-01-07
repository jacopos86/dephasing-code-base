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
echo "DELTA environment setup complete."
