from pydephasing.input_parameters import p
from pydephasing.log import log
from pydephasing.input_parser import parser
from pydephasing.T2_calc import (T2_eval_from_integ_class, 
                T2_eval_fit_model_dyn_inhom_class, 
                T2_eval_fit_model_dyn_homo_class,
                T2_eval_from_integ_homo_class,
                T2_eval_from_integ_inhom_class)
#
#   here we define the T2 calculation handler
#
def set_T2_calc_handler():
    if p.fit_acf_oft and p.integ_acf_oft:
        log.error("fit_acf_oft and integ_acf_oft cannot be both True")
    if not p.fit_acf_oft and not p.integ_acf_oft:
        log.error("fit_acf_oft and integ_acf_oft cannot be both False")
    # if fit acf
    if p.time_resolved:
        if p.fit_acf_oft:
            if not p.deph and not p.relax:
                pass
            else:
                calc_type2 = parser.parse_args().ct2
                if p.ACF_FIT:
                    if calc_type2 == "homo":
                        return T2_eval_fit_model_dyn_homo_class()
                    elif calc_type2 == "inhomo" or calc_type2 == "full":
                        return T2_eval_fit_model_dyn_inhom_class()
                elif p.ACF_INTEG:
                    if calc_type2 == "homo":
                        return T2_eval_from_integ_homo_class()
                    elif calc_type2 == "inhomo" or calc_type2 == "full":
                        return T2_eval_from_integ_inhom_class()
                else:
                    log.error("T2_extract_method : fit / integ")
        elif p.integ_acf_oft:
            return T2_eval_from_integ_class()