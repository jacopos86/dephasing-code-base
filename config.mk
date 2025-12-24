# ===================
#  Project paths
# ===================

ROOT := $(shell pwd)
VENV := $(ROOT)/pydeph

PYTHON_VERSION := python3

# ===================
# Build mode
# ===================

BUILD_MODE ?= local

# ===================
# pycuda install
# ===================

INSTALL_PYCUDA ?= 0

# ===================
#  PETSc variables
# ===================

PETSC_DIR ?=
PETSC_ARCH ?=
