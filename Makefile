VENV = pydeph
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
EXAMPLES_DIR = EXAMPLES
EXAMPLES_TAR_FILE = EXAMPLES.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"

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
	if [ -d $(VENV) ] ; \
	then \
		rm -rf $(VENV) ; \
	fi ; \
	if [ -f ./config.yml ] ; \
	then \
		rm ./config.yml ; \
	fi ;
test :
	cd ./tests 
	python -m unittest test_unit