# ===================
#  Load config
# ===================

include config.mk

# ===================
#  Local python
# ===================

PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# ===================
#  DATA FILES
# ===================

EXAMPLES_TAR_FILE := $(ROOT)/EXAMPLES.tar.gz
TESTS_3_TAR_FILE := $(ROOT)/TESTS_3.tar.gz
EXAMPLES_URL := "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
TESTS_3_URL := "https://drive.google.com/file/d/1Vv_xmpivm8p0vjsTG0MIlB2Th7GykTQk/view?usp=drive_link"

# ===================
# TEST DIR
# ===================

UNIT_TEST_DIR := $(ROOT)/pydephasing/unit_tests
NP_MAX := 2

# ===================
# Skipping downloads
# ===================

DOWNLOAD_EXAMPLES ?= 1
DOWNLOAD_TESTS3 ?= 1


configure : $(ROOT)/requirements.txt
	$(PYTHON_VERSION) -m venv $(VENV)
	echo 'export PYTHONPATH="$(ROOT):$$PYTHONPATH"' >> $(VENV)/bin/activate
	. $(VENV)/bin/activate && \
	$(PIP) install --upgrade pip setuptools wheel && \
	$(PIP) install --only-binary=phonopy phonopy || $(PIP) install --no-build-isolation phonopy && \
	$(PIP) install -r $(ROOT)/requirements.txt --no-deps && \
	if [ "$(BUILD_MODE)" = "nersc" ]; then \
		if [ -z "$$PETSC_DIR" ] || [ -z "$$PETSC_ARCH" ]; then \
			echo "ERROR: PETSC_DIR and PETSC_ARCH must be set"; exit 1; \
		fi; \
		PETSC_DIR=$$PETSC_DIR PETSC_ARCH=$$PETSC_ARCH $(PIP) install petsc4py; \
		MPICC=mpicc HDF5_MPI=ON $(PIP) install --no-binary=h5py h5py; \
		MPICC=mpicc $(PIP) install --no-binary=mpi4py mpi4py; \
	else \
		$(PIP) install petsc petsc4py mpi4py; \
		if [ "$$INSTALL_PYCUDA" = "1" ]; then \
			echo "Installing pycuda ..."; \
			$(PIP) install pycuda; \
		else \
			echo "Skipping pycuda (set INSTALL_PYCUDA=1 to enable)"; \
		fi; \
	fi;
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
	. $(VENV)/bin/activate ; \
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_1.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest -p no:warnings $(UNIT_TEST_DIR)/test_2.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_3.py
	PYDEPHASING_TESTING=1 $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_5.py
	@for np in $$(seq 1 $(NP_MAX)); do \
		PYDEPHASING_TESTING=1 mpirun -np $$np $(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_6.py; \
	done
