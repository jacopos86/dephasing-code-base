from utilities.input_parser import parser
from pydephasing.input_parameters import preproc_data_input, static_data_input, linear_resp_input, real_time_input
# input parameters object
p = None
calc_type1 = parser.parse_args().ct1[0]
if calc_type1 == "init":
    p = preproc_data_input()
else:
    ct1 = parser.parse_args().ct1[0]
    ct2 = parser.parse_args().ct2
    if ct1 == "LR":
        if ct2 == "inhomo" or ct2 == "homo" or ct2 == "full":
            p = linear_resp_input()
        elif ct2 == "stat" or ct2 == "statdd":
            p = static_data_input()
    elif ct1 == "RT":
        p = real_time_input()
p.sep = "*"*94