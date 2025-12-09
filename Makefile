ROOT = $(shell pwd)
VENV = $(ROOT)/pydeph
PYTHON_VERSION = python3
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# DATA FILE
EXAMPLES_TAR_FILE = $(ROOT)/EXAMPLES.tar.gz
TESTS_3_TAR_FILE = $(ROOT)/TESTS_3.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
TESTS_3_URL = "https://drive.google.com/file/d/1Vv_xmpivm8p0vjsTG0MIlB2Th7GykTQk/view?usp=drive_link"

# TEST DIR
UNIT_TEST_DIR = $(ROOT)/pydephasing/unit_tests
NP_MAX := 2

# Options for skipping large downloads
DOWNLOAD_EXAMPLES ?= 1
DOWNLOAD_TESTS3 ?= 1

# Optional override for CI
REQUIREMENTS_OVERRIDE ?=

configure : $(ROOT)/requirements.txt $(ROOT)/requirements_GPU.txt
	python3 -m venv $(VENV); \
	. $(VENV)/bin/activate; \
	$(PIP) install --upgrade pip setuptools wheel; \
	\
	# Detect if running on the HPC (Cray / Lmod environment)
	if command -v module >/dev/null 2>&1 && module avail 2>&1 | grep -q "petsc/3.23.4-cuda-gcc13.3.1"; then \
		echo "==> HPC environment detected. Loading PETSc module..."; \
		module load petsc/3.23.4-cuda-gcc13.3.1; \
		\
		echo "==> Installing petsc4py matching HPC PETSc..."; \
		$(PIP) install petsc4py==3.23.4; \
		\
		echo "==> Installing all other requirements (ignoring petsc & petsc4py if present)..."; \
		grep -v '^petsc$$' $(ROOT)/requirements.txt | grep -v '^petsc4py$$' > /tmp/req_cpu.txt; \
		grep -v '^petsc$$' $(ROOT)/requirements_GPU.txt | grep -v '^petsc4py$$' > /tmp/req_gpu.txt; \
		if ! command -v nvcc >/dev/null 2>&1; then \
			$(PIP) install -r /tmp/req_cpu.txt; \
		else \
			$(PIP) install -r /tmp/req_gpu.txt; \
		fi; \
	\
	else \
		echo "==> Non-HPC environment (MacOS / workstation). Installing normally."; \
		$(PIP) install -r $(ROOT)/requirements.txt; \
	fi
build :
	. $(VENV)/bin/activate ; \
	if [ "$(DOWNLOAD_EXAMPLES)" = "1" ]; then \
		if [ ! -f $(EXAMPLES_TAR_FILE) ] ; then \
			echo "Downloading EXAMPLES..."; \
			gdown --fuzzy $(EXAMPLES_URL) ; \
		fi ; \
	else \
		echo "Skipping EXAMPLES download"; \
	fi
	
	if [ "$(DOWNLOAD_TESTS3)" = "1" ]; then \
		if [ ! -f $(TESTS_3_TAR_FILE) ] ; then \
			echo "Downloading TESTS_3..."; \
			gdown --fuzzy $(TESTS_3_URL) ; \
		fi ; \
	else \
		echo "Skipping TESTS_3 download"; \
	fi
	./build.sh
install :
	. $(VENV)/bin/activate ; \
	export PETSC_DIR=$${PETSC_DIR:-$(PETSC_DIR)} ; \
	export PETSC_ARCH=$${PETSC_ARCH:-$(PETSC_ARCH)} ; \
	$(PIP) install --no-cache-dir --no-build-isolation .
.PHONY :
	clean
clean :
	@echo "Cleaning project ..."
	find $(ROOT) -name '__pycache__' -type d -exec rm -rf {} +
	rm -rf $(ROOT)/pydephasing/*~ ; \
	if [ -d $(ROOT)/build ] ; \
	then \
		rm -rf $(ROOT)/build ; \
	fi ; \
	if [ -d $(VENV) ] ; \
	then \
		rm -rf $(VENV) ; \
	fi ; \
	if [ -f $(ROOT)/config.yml ] ; \
	then \
		rm $(ROOT)/config.yml ; \
	fi ;
test :
	. $(VENV)/bin/activate ; \
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_1.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest -p no:warnings $(UNIT_TEST_DIR)/test_2.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_3.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_4.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_5.py
	@for np in $$(seq 1 $(NP_MAX)); do \
		PYDEPHASING_TESTING=1 srun -n $$np $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_6.py; \
	done

	
