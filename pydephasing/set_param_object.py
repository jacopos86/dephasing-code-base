import os
from pydephasing.utilities.input_parser import parser
from pydephasing.utilities.log import log
from pydephasing.input_parameters import (
    preproc_data_input, 
    static_data_input, 
    linear_resp_input, 
    real_time_SQ_input,
    Q_real_time_input,
    real_time_JDFTx_input,
    real_time_VASP_input
)

# parameters proxy class

class param_proxy:
    def __init__(self):
        self._real_p = None

    def set_input_arguments(self):
        is_testing = os.getenv('PYDEPHASING_TESTING') == '1'
        if is_testing is True:
            args = parser.parse_args(args=[])
        else:
            args = parser.parse_args()
        return args
    
    def _init(self):
        # input parameters object initialization
        args = self.set_input_arguments()
        co = parser.parse_args().co[0]
        ct1 = args.ct1[0]
        if co == 'spin-qubit' or co == 'energy-lw':
            if ct1 == "init":
                self._real_p = preproc_data_input()
            else:
                ct2 = args.ct2
                if ct1 == "LR":
                    if ct2 == "inhomo" or ct2 == "homo" or ct2 == "full":
                        self._real_p = linear_resp_input()
                    elif ct2 == "stat" or ct2 == "statdd":
                        self._real_p = static_data_input()
                    else:
                        log.error(f"Unknown ct2 value: {ct2!r}")
                elif ct1 == "RT":
                    self._real_p = real_time_SQ_input()
                elif ct1 == "QUANTUM":
                    self._real_p = Q_real_time_input()
                else:
                    log.error(f"Unknown ct1 value: {ct1!r}")
        elif co == 'elec-sys':
            if ct1 == "RT":
                ct2 = args.ct2
                if ct2 == "jdftx":
                    self._real_p = real_time_JDFTx_input()
                elif ct2 == "vasp":
                    self._real_p = real_time_VASP_input()
                elif ct2 == "MODEL":
                    self._real_p = real_time_JDFTx_input()
                else:
                    log.error(f"Unknown ct2 value: {ct2!r}")
            elif ct1 == "LR":
                if ct2 == "vasp":
                    self._real_p = linear_resp_input()
                else:
                    log.error(f"Unknown ct2 value: {ct2!r}")
            else:
                log.error(f"ONLY RT IMPLEMENTED")
        if self._real_p is None:
            log.error(f"Failed to initialize parameter object with ct1={ct1!r}, ct2={ct2!r}")
        self._real_p.sep = "*"*94

    def __getattr__(self, attr):
        if self._real_p is None:
            self._init()
        return getattr(self._real_p, attr)
    
p = param_proxy()