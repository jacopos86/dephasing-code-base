from pydephasing.input_parser import parser
from pydephasing.input_parameters import preproc_data_input, static_data_input, dynamical_data_input
# input parameters object
p = None
calc_type1 = parser.parse_args().ct1[0]
if calc_type1 == "init":
    p = preproc_data_input()
else:
    ct2 = parser.parse_args().ct2
    if ct2 == "inhomo" or ct2 == "homo" or ct2 == "full":
        p = dynamical_data_input()
    elif ct2 == "stat" or ct2 == "statdd":
        p = static_data_input()
p.sep = "*"*94