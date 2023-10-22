QEpath = /home/jacopo/Documents/q-e-qe-6.5

build :
	pip install -r requirements.txt
configure :
	./configure.sh
install :
	python setup.py install
.PHONY :
	clean
clean :
	rm -rf ./pydephasing/*~ ./pydephasing/__pycache__ ./build/lib/pydephasing/* ./__pycache__ ./config.yml
test :
	cd ./tests; python -m unittest test_unit; rm -r __pycache__
QE2pydeph :
	cd $(QEpath); pwd
