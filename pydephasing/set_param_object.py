from pydephasing.input_parser import parser
from pydephasing.input_parameters import preproc_data_input, static_data_input, dynamical_data_input
# input parameters object
p = None
calc_type1 = parser.parse_args().ct1[0]
if calc_type1 == "init":
    p = preproc_data_input()
else:
    deph_type = parser.parse_args().typ
    if deph_type == "stat" or deph_type == "statdd":
        p = static_data_input()
    elif deph_type == "relax" or deph_type == "deph":
        p = dynamical_data_input()
p.sep = "*"*94