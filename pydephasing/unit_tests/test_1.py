import argparse
import os
from utilities.log import log
from pydephasing.input_parameters import preproc_data_input

TESTS_DIR = "TESTS"

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

def test_program_parse():
    yml_inp = str(os.getcwd())+'/'+str(TESTS_DIR)+"/1/input.yml"
    argv_list = ["--yml_inp", yml_inp]
    args = parse_args(argv_list)
    yml_inp_parse = args.yml_inp
    assert yml_inp == yml_inp_parse

def test_parameters():
    yml_inp = str(os.getcwd()) + '/' + str(TESTS_DIR) + "/1/input.yml"
    pobj = read_parameters(yml_inp)
    assert pobj.displ_poscar_dir[0] == str(os.getcwd())+'/'+str(TESTS_DIR)+"/1/DISPLACEMENT-FILES-001"
    assert pobj.displ_outcar_dir[0] == str(os.getcwd())+'/'+str(TESTS_DIR)+"/1/DISPL-001"
    assert pobj.atoms_displ[0][0] == 0.01
    assert pobj.atoms_displ[0][1] == 0.01
    assert pobj.atoms_displ[0][2] == 0.01
    assert pobj.defect_index == 0
    assert pobj.max_dab == 2.7