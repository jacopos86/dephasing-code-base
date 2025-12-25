#!/bin/bash
# -----------------------------
# NERSC HPC environment setup
# -----------------------------

# Restore system defaults first
source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh

# Load required modules
module load python

# -----------------------------
# Build mode and GPU flag
# -----------------------------
export BUILD_MODE=nersc
export INSTALL_PYCUDA=0
export PETSC_DIR=/pscratch/sd/j/jsimoni/PYDEPHASING/petsc
export PETSC_ARCH=arch-linux-c-opt

# -----------------------------
# PETSc installation
# -----------------------------

if [ ! -d "$PETSC_DIR" ]; then
	echo "PETSc not found at $PETSC_DIR"
	echo "Installing PETSc..."
	
	git clone -b release https://gitlab.com/petsc/petsc.git $PETSC_DIR
	cd $PETSC_DIR	

	./configure \
		--PETSC_ARCH="$PETSC_ARCH" \
		--with-cc=mpicc \
		--with-cxx=mpicxx \
		--with-fc=mpif90 \
		--with-shared-libraries=1
	make all
else
    echo "Using existing PETSc at $PETSC_DIR"
fi

# -----------------------------
# Verify MPI and PETSc
# -----------------------------
echo "MPI compiler:"
which mpicc
mpicc -show
echo "Check PETSc header:"
ls $PETSC_DIR/include/petsc.h 2>/dev/null || echo "PETSc header not found."
# -----------------------------
# Finished
# -----------------------------
echo "NERSC environment setup complete."
