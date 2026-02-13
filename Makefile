# ===================
#  Load config
# ===================

include config.mk

# ===================
#  export variables
#  Python
# ===================

export ROOT
export GPU_ACTIVE
export GRID_SIZE
export BLOCK_SIZE
export TESTS_DIR
export PYDEPHASING_TESTING
export LOG_LEVEL
export COLOR_LOG
export LOG_FILE

# ========================
# Base conda dependencies
# ========================

CONDA_BASE_DEPS = \
  python=3.11 \
  numpy \
  scipy \
  matplotlib \
  mpi4py \
  h5py \
  pip
  
# ================================
# Generate conda-environment.yml
# ================================

environment:
	@echo "Generating $(CONDA_ENV_FILE)"
	@rm -f $(CONDA_ENV_FILE)
	@echo "name: $(CONDA_ENV_NAME)" >> $(CONDA_ENV_FILE)
	@echo "channels:" >> $(CONDA_ENV_FILE)
	@echo "  - conda-forge" >> $(CONDA_ENV_FILE)
	@echo "dependencies:" >> $(CONDA_ENV_FILE)
	@for pkg in $(CONDA_BASE_DEPS); do \
		echo "  - $$pkg" >> $(CONDA_ENV_FILE); \
	done
ifeq ($(GPU_ACTIVE),1)
	@echo "  - \"petsc=*=*cuda*\"" >> $(CONDA_ENV_FILE)
	@echo "  - \"petsc4py=*=*cuda*\"" >> $(CONDA_ENV_FILE)
else
	@echo "  - \"petsc=*=*complex*\"" >> $(CONDA_ENV_FILE)
	@echo "  - \"petsc4py=*=*complex*\"" >> $(CONDA_ENV_FILE)
endif
ifeq ($(INSTALL_PYCUDA),1)
	@echo "  - pycuda" >> $(CONDA_ENV_FILE)
endif
	@echo "  - pip:" >> $(CONDA_ENV_FILE)
	@echo "      - -r $(ROOT)/dependencies/requirements.txt" >> $(CONDA_ENV_FILE)
	@echo "      - -e $(ROOT)" >> $(CONDA_ENV_FILE)

# ===================
#  configuration
# ===================

configure : $(CONDA_ENV_FILE)
	@echo "Checking conda environment $(CONDA_ENV_NAME)..."
	@if ! $(CONDA) env list | grep -qw $(CONDA_ENV_NAME); then \
		echo "Creating conda environment $(CONDA_ENV_NAME) ..."; \
		echo "$(CONDA_ENV_FILE)"; \
		$(CONDA) env create -f $(CONDA_ENV_FILE); \
	else \
		echo "Updating existing conda environment $(CONDA_ENV_NAME) ..."; \
		$(CONDA) env update -f $(CONDA_ENV_FILE) --prune; \
	fi
	
# ===================
#  build section
# ===================
	
build :
	@echo "Running build inside conda environment $(CONDA_ENV_NAME)" \
	mkdir -p "$(TESTS_DIR)" && \
	if [ "$(DOWNLOAD_EXAMPLES)" = "1" ]; then \
		if [ ! -f $(TESTS_12_TAR_FILE) ] ; then \
			echo "Downloading EXAMPLES..."; \
			$(CONDA) run -n $(CONDA_ENV_NAME) gdown $(EXAMPLES_FILE_ID) -O "$(TESTS_12_TAR_FILE)"; \
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
			$(CONDA) run -n $(CONDA_ENV_NAME) gdown $(TESTS_3_FILE_ID) -O "$(TESTS_3_TAR_FILE)"; \
			if [ ! -f $(TESTS_3_TAR_FILE) ]; then \
				echo "ERROR: TESTS_3 download failed!"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "Skipping TESTS_3 download"; \
	fi; \
	cd $(ROOT) && \
	./build.sh "$(LOG_LEVEL)" "$(COLOR_LOG)" "$(LOG_FILE)" "$(INSTALL_PYCUDA)" "$(BLOCK_SIZE)" "$(GRID_SIZE)" "$(BUILD_TESTS)" "$(TESTS_12_TAR_FILE)" "$(TESTS_3_TAR_FILE)" 
	
# ===================
#  install section
# ===================

install :
	@echo "Installing package into conda environment $(CONDA_ENV_NAME)"
	@$(CONDA) run -n $(CONDA_ENV_NAME) python -m pip install -e .

.PHONY :
	clean

# ===================
#  clean section
# ===================

clean :
	@echo "Cleaning project ..."
	# remove pycache and temporary files
	find $(ROOT) -name '__pycache__' -type d -exec rm -rf {} +
	rm -rf $(ROOT)/pydephasing/*~ ; \
	if [ -d $(ROOT)/build ] ; \
	then \
		rm -rf $(ROOT)/build ; \
	fi ; \
	# remove conda environment if it exists
	@if $(CONDA) env list | grep -qw $(CONDA_ENV_NAME); then \
		echo "Removing conda environment $(CONDA_ENV_NAME) ..."; \
		$(CONDA) env remove -n $(CONDA_ENV_NAME); \
	fi

# ===================
#  test section
# ===================

test :
	@echo "Running tests in conda environment $(CONDA_ENV_NAME)"
	@set -e; \
	PYTEST="$(CONDA) run -n $(CONDA_ENV_NAME) python -m pytest"; \
	PYDEPHASING_TESTING=1 $$PYTEST $(UNIT_TEST_DIR)/test_1.py; \
	PYDEPHASING_TESTING=1 $$PYTEST $(UNIT_TEST_DIR)/test_3.py; \
	PYDEPHASING_TESTING=1 $$PYTEST $(UNIT_TEST_DIR)/test_5.py; \
	MPI_CMD="$(CONDA) run -n $(CONDA_ENV_NAME) $(MPI_LAUNCHER) -np"; \
	for np in $$(seq 1 $(NP_MAX)); do \
		PYDEPHASING_TESTING=1 $$MPI_CMD $$np python -m pytest $(UNIT_TEST_DIR)/test_6.py; \
	done; \
	PYDEPHASING_TESTING=1 $$PYTEST $(UNIT_TEST_DIR)/test_7.py
