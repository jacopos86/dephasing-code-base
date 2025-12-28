# ===================
#  Load config
# ===================

include config.mk

# ===================
#  Local python
# ===================

PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

configure : $(ROOT)/dependencies/requirements.txt $(ROOT)/dependencies/requirements_GPU.txt $(ROOT)/dependencies/requirements-HPC.txt $(ROOT)/dependencies/requirements-HPC-Delta.txt $(ROOT)/dependencies/requirements-HPC_GPU.txt
	$(PYTHON_VERSION) -m venv $(VENV)
	echo 'export PYTHONPATH="$(ROOT):$$PYTHONPATH"' >> $(VENV)/bin/activate
	. $(VENV)/bin/activate && \
	$(PIP) install --upgrade pip setuptools wheel && \
	if [ "$(BUILD_MODE)" = "nersc" ]; then \
		if [ -z "$$PETSC_DIR" ] || [ -z "$$PETSC_ARCH" ]; then \
			echo "ERROR: PETSC_DIR and PETSC_ARCH must be set"; exit 1; \
		fi; \
		$(PIP) install -r $(ROOT)/dependencies/requirements-HPC.txt; \
		PETSC_DIR=$$PETSC_DIR PETSC_ARCH=$$PETSC_ARCH $(PIP) install petsc4py; \
		CC=cc MPICC=mpicc HDF5_MPI=ON $(PIP) install --no-binary=h5py h5py; \
		MPICC=mpicc $(PIP) install --no-binary=mpi4py mpi4py; \
	elif [ "$(BUILD_MODE)" = "delta" ]; then \
		$(PIP) install -r $(ROOT)/dependencies/requirements-HPC-Delta.txt; \
		module load petsc/3.23.4-cuda-gcc13.3.1; \
		$(PIP) install petsc4py==3.23.4; \
		MPICC=cc $(PIP) install --no-binary=mpi4py mpi4py; \
	else \
		$(PIP) install petsc; \
		if [ "$$INSTALL_PYCUDA" = "1" ]; then \
			echo "Installing pycuda ..."; \
			$(PIP) install -r $(ROOT)/dependencies/requirements_GPU.txt; \
		else \
			echo "Skipping pycuda (set INSTALL_PYCUDA=1 to enable)"; \
			$(PIP) install -r $(ROOT)/dependencies/requirements.txt; \
		fi; \
	fi;
build :
	. $(VENV)/bin/activate && \
	mkdir -p "$(TESTS_DIR)" && \
	if [ "$(DOWNLOAD_EXAMPLES)" = "1" ]; then \
		if [ ! -f $(TESTS_12_TAR_FILE) ] ; then \
			echo "Downloading EXAMPLES..."; \
			gdown $(EXAMPLES_FILE_ID) -O "$(TESTS_12_TAR_FILE)"; \
			if [ ! -f $(TESTS_12_TAR_FILE) ]; then \
				echo "ERROR: EXAMPLES download failed!"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "Skipping EXAMPLES download"; \
	fi; \
	if [ "$(DOWNLOAD_TESTS3)" = "1" ]; then \
		if [ ! -f $(TESTS_3_TAR_FILE) ]; then \
			echo "Downloading TESTS_3..."; \
			gdown $(TESTS_3_FILE_ID) -O "$(TESTS_3_TAR_FILE)"; \
			if [ ! -f $(TESTS_3_TAR_FILE) ]; then \
				echo "ERROR: TESTS_3 download failed!"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "Skipping TESTS_3 download"; \
	fi; \
	cd $(ROOT) && \
	./build.sh "$(LOG_LEVEL)" "$(COLOR_LOG)" "$(LOG_FILE)" "$(INSTALL_PYCUDA)" "$(BLOCK_SIZE)" "$(GRID_SIZE)" $(BUILD_TESTS) "$(TESTS_12_TAR_FILE)" "$(TESTS_3_TAR_FILE)" 
install :
	. $(VENV)/bin/activate ; \
	$(PIP) install .
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
	. $(VENV)/bin/activate && \
	set -e && \
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_1.py && \
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest -p no:warnings $(UNIT_TEST_DIR)/test_2.py && \
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_3.py && \
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_5.py && \
	if [ "$(BUILD_MODE)" != "nersc" ]; then \
		for np in $$(seq 1 $(NP_MAX)); do \
			PYDEPHASING_TESTING=1 mpirun -np $$np $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_6.py; \
		done; \
	else \
		echo "Skipping MPI tests"; \
	fi
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_7.py
