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

# ===================
#  TESTS dir.
# ===================

TESTS_DIR := $(ROOT)/TESTS

# ===================
#   tar.gz files url
# ===================

EXAMPLES_URL := "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
TESTS_3_URL := "https://drive.google.com/file/d/1Vv_xmpivm8p0vjsTG0MIlB2Th7GykTQk/view?usp=drive_link"

# ===================
# Skipping downloads
# ===================

DOWNLOAD_EXAMPLES ?= 1
DOWNLOAD_TESTS3 ?= 1

# ===================
#  Log level
# ===================

LOG_LEVEL := INFO
