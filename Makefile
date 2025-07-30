VENV = pydeph
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
ROOT = $(shell pwd)
EXAMPLES_TAR_FILE = EXAMPLES.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
UNIT_TEST_DIR = pydephasing/unit_tests
TESTS_DIR = TESTS

configure : requirements.txt requirements_GPU.txt
	python3 -m venv $(VENV); \
	. $(VENV)/bin/activate;
ifeq (, $(shell which nvcc))
	$(PIP) install -r requirements.txt
else
	$(PIP) install -r requirements_GPU.txt
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
	rm -rf ./pydephasing/*~ ; \
	if [ -d ./pydephasing/__pycache__ ] ; \
	then \
		rm -rf ./pydephasing/__pycache__ ; \
	fi ; \
	if [ -d ./build ] ; \
	then \
		rm -rf ./build ; \
	fi ; \
	if [ -d ./__pycache__ ] ; \
	then \
		rm -rf ./__pycache__ ; \
	fi ; \
	if [ -d ./pydephasing/unit_tests/__pycache__ ] ; \
	then \
		rm -rf ./pydephasing/unit_tests/__pycache__ ; \
	fi ; \
	if [ -d ./pydephasing/common/__pycache__ ] ; \
	then \
		rm -rf ./pydephasing/common/__pycache__ ; \
	fi ; \
	if [ -d $(VENV) ] ; \
	then \
		rm -rf $(VENV) ; \
	fi ; \
	if [ -f ./config.yml ] ; \
	then \
		rm ./config.yml ; \
	fi ;
test :
	. $(VENV)/bin/activate ; \
	pytest
