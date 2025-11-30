ROOT = $(shell pwd)
VENV = $(ROOT)/pydeph
PYTHON_VERSION = python3
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
EXAMPLES_TAR_FILE = $(ROOT)/EXAMPLES.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
UNIT_TEST_DIR = $(ROOT)/pydephasing/unit_tests
NP_MAX := 2

configure : $(ROOT)/requirements.txt $(ROOT)/requirements_GPU.txt
	$(PYTHON_VERSION) -m venv $(VENV); \
	. $(VENV)/bin/activate;
	$(PIP) install --upgrade pip setuptools wheel
ifeq (, $(shell which nvcc))
	$(PIP) install -r $(ROOT)/requirements.txt
else
	$(PIP) install -r $(ROOT)/requirements_GPU.txt
endif
build :
	. $(VENV)/bin/activate; \
	if [ ! -f $(EXAMPLES_TAR_FILE) ] ; \
	then \
		gdown --fuzzy $(EXAMPLES_URL) ; \
	fi ; \
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
