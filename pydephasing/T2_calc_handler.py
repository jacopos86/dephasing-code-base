from pydephasing.input_parameters import p
#
#   here we define the T2 calculation handler
#
def set_T2_calc_handler():
    # if fit acf
    if p.time_resolved:
        print(p.fit_acf_oft, p.integ_acf_oft)
        if p.fit_acf_oft:
            if not p.deph and not p.relax:
                pass
            else: 
                calc_type2 = parser.parse_args().ct2
                if calc_type2 == "homo":
                    return T2_eval_fit_model_dyn_class()
                elif calc_type2 == "inhomo" or calc_type2 == "full":
                    pass
        if p.integ_acf_oft:
            return T2_eval_from_integ_class()