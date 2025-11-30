from pydephasing.electr_ph_coupling import eph_deform_pot_model


#
def compute_RT_electronic_dephas(MODEL=False):
    # if run model
    if MODEL:
        eph = eph_deform_pot_model()
    else:
        eph = set_electron_phonon_coupling()