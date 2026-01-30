# ===================
#  Project paths
# ===================

ROOT := $(shell pwd)
CONDA := conda
CONDA_ENV_NAME := pydeph
CONDA_ENV_FILE := $(ROOT)/conda-environment.yml

# ===================
# Build mode
# ===================

BUILD_MODE ?= local

# ===================
# pycuda install
# ===================

INSTALL_PYCUDA ?= 0
GPU_ACTIVE ?= 0
GRID_SIZE ?=
BLOCK_SIZE ?=

# ===================
# MPI launcher
# ===================

MPI_LAUNCHER ?= mpirun
ifeq ($(BUILD_MODE),nersc)
	MPI_LAUNCHER := srun
endif

# ===================
#  unit tests
# ===================

NP_MAX ?= 2
UNIT_TEST_DIR := $(ROOT)/pydephasing/unit_tests
PYDEPHASING_TESTING ?= 0

# ===================
#  TESTS dir.
# ===================

BUILD_TESTS ?= "1 2"
TESTS_DIR := $(ROOT)/TESTS
TESTS_12_TAR_FILE := $(ROOT)/EXAMPLES.tar.gz
TESTS_3_TAR_FILE := $(ROOT)/TESTS_3.tar.gz

# ===================
#   tar.gz files url
# ===================

EXAMPLES_URL := "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
EXAMPLES_FILE_ID := 1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9
TESTS_3_URL := "https://drive.google.com/file/d/1Vv_xmpivm8p0vjsTG0MIlB2Th7GykTQk/view?usp=drive_link"
TESTS_3_FILE_ID := 1Vv_xmpivm8p0vjsTG0MIlB2Th7GykTQk

# ===================
# Skipping downloads
# ===================

DOWNLOAD_EXAMPLES ?= 1
DOWNLOAD_TESTS3 ?= 1

# ===================
#  Log level
# ===================

LOG_LEVEL ?= INFO
COLOR_LOG ?= 1
LOG_FILE ?= out.log
