ROOT = $(shell pwd)
VENV = $(ROOT)/pydeph
PYTHON_VERSION = python3
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
EXAMPLES_TAR_FILE = $(ROOT)/EXAMPLES.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
UNIT_TEST_DIR = $(ROOT)/pydephasing/unit_tests

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
	rm -rf $(ROOT)/pydephasing/*~ ; \
	if [ -d $(ROOT)/pydephasing/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/pydephasing/__pycache__ ; \
	fi ; \
	if [ -d $(ROOT)/build ] ; \
	then \
		rm -rf $(ROOT)/build ; \
	fi ; \
	if [ -d $(ROOT)/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/__pycache__ ; \
	fi ; \
	if [ -d $(UNIT_TEST_DIR)/__pycache__ ] ; \
	then \
		rm -rf $(UNIT_TEST_DIR)/__pycache__ ; \
	fi ; \
	if [ -d $(ROOT)/pydephasing/common/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/pydephasing/common/__pycache__ ; \
	fi ; \
	if [ -d $(ROOT)/pydephasing/real_time/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/pydephasing/real_time/__pycache__ ; \
	fi ; \
	if [ -d $(ROOT)/pydephasing/parallelization/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/pydephasing/parallelization/__pycache__ ; \
	fi ; \
	if [ -d $(ROOT)/pydephasing/spin_model/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/pydephasing/spin_model/__pycache__ ; \
	fi ; \
	if [ -d $(ROOT)/pydephasing/quantum/__pycache__ ] ; \
	then \
		rm -rf $(ROOT)/pydephasing/quantum/__pycache__ ; \
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
	$(PYTHON) -m pytest $(UNIT_TEST_DIR)/test_1.py
	$(PYTHON) -m pytest -p no:warnings $(UNIT_TEST_DIR)/test_2.py
