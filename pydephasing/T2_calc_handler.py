from pydephasing.set_param_object import p
from utilities.log import log
from utilities.input_parser import parser
from pydephasing.T2_calc import (T2_eval_freq_homo_class,
                T2_eval_freq_inhom_class,
                T2_eval_fit_model_dyn_inhom_class, 
                T2_eval_fit_model_dyn_homo_class,
                T2_eval_from_integ_homo_class,
                T2_eval_from_integ_inhom_class,
                T2_eval_static_class,
                T2_eval_dyndec_class)
#
#   here we define the T2 calculation handler
#
def set_T2_calc_handler():
    # calc type 2 : homo/inhomo
    calc_type2 = parser.parse_args().ct2
    #
    # dynamical calculation branch
    #
    if p.relax or p.deph:
        if p.ACF_FIT and p.ACF_INTEG:
            log.error("ACF_FIT and ACF_INTEG cannot be both True")
        # if time resolved acf
        if p.time_resolved:
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
        # w resolved
        elif p.w_resolved:
            if calc_type2 == "homo":
                return T2_eval_freq_homo_class()
            elif calc_type2 == "inhomo" or calc_type2 == "full":
                return T2_eval_freq_inhom_class()
    else:
        # static calculation
        if calc_type2 == "inhomo":
            if p.dyndec:
                return T2_eval_dyndec_class()
            else:
                return T2_eval_static_class()