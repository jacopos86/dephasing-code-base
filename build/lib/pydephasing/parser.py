import argparse
import sys
# set up parser
# make parser global to read parameters
parser = argparse.ArgumentParser()
parser.add_argument("calcType1")
parser.add_argument("calcType2")
parser.add_argument("dephType")
parser.add_argument("yml_input")