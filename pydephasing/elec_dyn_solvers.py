# This is the main subroutine
# for non Markovian calculation of electronic systems
#
from pydephasing.set_param_object import p
from parallelization.mpi import mpi
from utilities.log import log
from wannier_interface.wannier import Wannier
from pydephasing.phonons_module import JDFTxPhonons
from pydephasing.set_structs import JDFTxStruct

def solve_elec_dyn_VASP_data():
    '''
    use VASP data to perform RT dynamics
    '''






def solve_elec_dyn_JDFTx_data():
    '''
    use JDFTx data to perform RT dynamics
    '''
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t PERFORM REAL TIME DYNAMICS")
        log.info("\t COLLECT DATA FROM JDFTx CALCULATION")
        log.info("\n")
        log.info("\t " + p.sep)
    # set electronic structure
    elec_struct = JDFTxStruct(p.bnd_kpts_file, p.eigenv_file, p.dft_outfile)
    elec_struct.set_elec_parameters()
    # read Wannier data
    wan = Wannier(elec_struct, p.cellmap_file, p.wan_weights_file, p.wan_mlwfh_file)
    wan.plot_band_structure()
    if p.dynamical_mode[1] > 0:
        # read phonons data
        # if we need to compute e-ph interactions
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t COLLECT PHONONS INFORMATION")
            log.info("\t " + p.sep)
            log.info("\n")
        ph = JDFTxPhonons(p.ph_cellmap_file, p.ph_eigenv_file, p.ph_outfile)
        ph.read_ph_hamilt()
        ph.get_ph_supercell()