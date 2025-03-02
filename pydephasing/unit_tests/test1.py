import argparse
from pydephasing.log import log
from pydephasing.input_parameters import preproc_data_input

def parse_args(sys_args):
    parse = argparse.ArgumentParser(description="Command line program.")
    parse.add_argument("--yml_inp", type=str,
                       help="Enter input")
    return parse.parse_args(sys_args)

def read_parameters(yml_inp):
    pobj = preproc_data_input()
    # read data
    pobj.read_yml_data(yml_inp)
    return pobj

def run(pobj):
    # start test
    log.info("\n")
    log.info("\t " + pobj.sep)
    log.info("\t BUILD DISPLACED STRUCTS.")