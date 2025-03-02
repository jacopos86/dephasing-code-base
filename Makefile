VENV = pydeph
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
ROOT = $(shell pwd)
EXAMPLES_DIR = EXAMPLES
EXAMPLES_TAR_FILE = EXAMPLES.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"
UNIT_TEST_DIR = pydephasing/unit_tests
TESTS_DIR = TESTS

configure : requirements.txt
	python3 -m venv $(VENV); \
	. $(VENV)/bin/activate; \
	$(PIP) install -r requirements.txt 
build :
	. $(VENV)/bin/activate; \
	if [ ! -f $(EXAMPLES_TAR_FILE) ] ; \
	then \
		gdown --fuzzy $(EXAMPLES_URL) ; \
	fi ; \
	if [ ! -d $(EXAMPLES_DIR) ] ; \
	then \
		tar -xvzf $(EXAMPLES_TAR_FILE) ; \
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
	export TESTS=$(ROOT)/$(TESTS_DIR) ; \
	$(PYTHON) -m unittest -v $(UNIT_TEST_DIR)/unit_test1.py