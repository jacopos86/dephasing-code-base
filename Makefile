VENV = pydeph
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
EXAMPLES_FILE = EXAMPLES
EXAMPLES_TAR_FILE = EXAMPLES.tar.gz
EXAMPLES_URL = "https://drive.google.com/file/d/1ueLGCuRSZO-c1hwrCvhO913TyBTjkuP9/view?usp=sharing&confirm=t"

configure : requirements.txt
	python3 -m venv $(VENV); \
	. $(VENV)/bin/activate; \
	$(PIP) install -r requirements.txt 
build :
	. $(VENV)/bin/activate; \
	if [ ! -f $(EXAMPLES_FILE) ] ; \
	then \
		gdown --fuzzy $(EXAMPLES_URL) ; \
		tar -xvzf $(EXAMPLES_TAR_FILE) ; \
	fi ; \
	./build.sh
install :
.PHONY :
	clean
clean :
	deactivate; \
	rm -rf ./pydephasing/*~ 
	rm -rf ./pydephasing/__pycache__ 
	rm -rf ./build 
	rm -rf ./__pycache__
	rm -rf $(VENV) 
	rm ./config.yml
test :
	cd ./tests 
	python -m unittest test_unit