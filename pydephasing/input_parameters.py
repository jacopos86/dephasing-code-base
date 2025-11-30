#
#   This module defines all the
#   parameters needed in input for the
#   calculations
#
import os
import numpy as np
import yaml
from abc import ABC
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi
from pydephasing.common.phys_constants import THz_to_ev
from pydephasing.common.matrix_operations import norm_cmplxv
from pydephasing.utilities.input_parser import parser
from pydephasing.common.grids import set_temperatures
#
class data_input(ABC):
    # initialization
    def __init__(self):
        #######################################
        # general input parameters
        # working directory
        self.work_dir = ''
        # write directory
        self.write_dir = ''
        # unperturbed directory
        self.unpert_dirs = []
        # general GS directory - no for grads
        self.gs_data_dir = ''
        # grad info file
        self.grad_info = ''
        # dynamical decoupling F by default
        self.dyndec = False
        # sep.
        self.sep = ''
        #######################################
        # physical common parameters
        # time
        self.T  = None
        self.dt = None
        self.nt = None
        ######################################
        # hyperfine parameters
        # n. spins list
        self.nsp = 0
        self.nconf = 0
        self.B0 = None
        # FC core
        self.fc_core = True
        # nuclear spin random
        # orientation
        self.rnd_orientation = False
        ######################################
        # auto-correlation fitting parameters
        # nlags
        self.nlags = 0
        # dephasing func. param.
        self.N_df= 100000
        self.T_df= 0.05
        self.maxiter = 80000
        ######################################
        # initial wave function
        self.psi0 = None
    #
    # read yaml data
    def read_yml_common_data(self, data):
        # extract directories
        # path always with respect to working directory
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        if 'output_dir' in data:
            self.write_dir = data['output_dir']
            # create output directory
            if mpi.rank == mpi.root:
                isExist = os.path.isdir(self.write_dir)
                if not isExist:
                    # create new dir.
                    os.makedirs(self.write_dir)
            mpi.comm.Barrier()
        if 'unpert_dir' in data:
            self.gs_data_dir = self.work_dir + '/' + data['unpert_dir']
        # grad info file
        if 'grad_info_file' in data:
            self.grad_info = self.work_dir + '/' + data['grad_info_file']
        # time variables
        if 'T' in data:
            # in ps units
            if isinstance(data['T'], list):
                if len(data['T']) > 2:
                    log.error("\t ONLY TWO T VALUES : 1) NORMAL CALC. 2) PH/AT RESOLVED")
                self.T = float(data['T'][0])
            else:
                self.T = float(data['T'])
        if 'dt' in data:
            self.dt = float(data['dt'])
            # ps units
        if 'nt' in data:
            self.nt = int(data['nt'])
        # applied static magnetic field
        # magnitude -> aligned along spin quant. axis
        if 'B0' in data:
            self.B0 = np.array(data['B0'])
        else:
            self.B0 = np.array([0., 0., 0.])
        # time dependent field
        if 'Bt' in data:
            self.Bt = data['Bt']
        # units : Tesla
        # psi0 wfc
        if 'psi0' in data:
            self.psi0 = np.array(data['psi0'], dtype=np.complex128)
            nrm = norm_cmplxv(self.psi0)
            self.psi0 = self.psi0 / nrm
        # ---------------------------------------
        #    HFI calculation parameters
        # ---------------------------------------
        calc_type2 = parser.parse_args().ct2
        if calc_type2 == "inhomo":
            self.read_hfi_data(data)
        # deph. function parameters
        if 'maxiter' in data:
            self.maxiter = int(data['maxiter'])
        if 'Ndf' in data:
            self.N_df = int(data['Ndf'])
        if 'Tdf' in data:
            self.T_df = float(data['Tdf'])
        if 'nlags' in data:
            self.nlags = int(data['nlags'])
    # read HFI data
    def read_hfi_data(self, data):
        # read n. of configurations
        if 'nconfig' in data:
            self.nconf = int(data['nconfig'])
        # n. of spins for each config.
        if 'nspins' in data:
            self.nsp = int(data['nspins'])
        # fermi contact term
        if 'core' in data:
            if data['core'] == False:
                self.fc_core = False
        # check random spin
        # orientation
        if 'random_orientation' in data:
            self.rnd_orientation = data['random_orientation']

class dynamical_data_input(data_input):
    # initialization
    def __init__(self):
        super().__init__()
        # perturbed calculations directory
        self.displ_poscar_dir = []
        self.displ_2nd_poscar_dir = []
        # perturbed calculations outcars directory
        self.displ_outcar_dir = []
        self.displ_2nd_outcar_dir = []
        # atoms displacements
        self.atoms_displ = []
        self.atoms_2nd_displ = []
        # yaml pos file
        self.yaml_pos_file = None
        # hdf5 file
        self.hd5_eigen_file = None
        #
        # atom resolved
        #
        self.at_resolved = False
        #
        # phonon resolved
        #
        self.ph_resolved = False
        self.nphr = 0
        self.nwbn = 0
        self.phm_list = []
        #
        # zfs/hfi 2nd order correction grad.
        self.order_2_correct = False
        self.hessian = False
        # fraction of atoms preserved
        # in gradient calculation
        self.frac_kept_atoms = 1.
        ####################################
        # physical parameters
        # n. temperatures
        self.temperatures = None
        self.ntmp = 0
        # eta decay parameter (time resolved calc.)
        self.eta = 1e-5
        # in eV units
        # min. allowed phonon freq.
        self.min_freq = 0.0
        # THz units
    #
    # read yml data file
    def read_yml_data_dyn(self, data):
        #
        # reaad common shared data
        #
        self.read_yml_common_data(data)
        # displ. positions directory
        if 'displ_poscar_dir' in data:
            for d in data['displ_poscar_dir']:
                self.displ_poscar_dir.append(self.work_dir + '/' + d)
        # 2nd order
        if 'displ_2nd_poscar_dir' in data:
            for d in data['displ_2nd_poscar_dir']:
                self.displ_2nd_poscar_dir.append(self.work_dir + '/' + d)
        # outcars
        if 'displ_outcar_dir' in data:
            for d in data['displ_outcar_dir']:
                self.displ_outcar_dir.append(self.work_dir + '/' + d)
        if 'displ_2nd_outcar_dir' in data:
            for d in data['displ_2nd_outcar_dir']:
                self.displ_2nd_outcar_dir.append(self.work_dir + '/' + d)
        # yml pos file
        if 'yaml_pos_file' in data:
            self.yaml_pos_file = self.work_dir + '/' + data['yaml_pos_file']
        # ph. data file
        if 'hd5_eigen_file' in data:
            self.hd5_eigen_file = self.work_dir + '/' + data['hd5_eigen_file']
        #
        # atom resolved
        if 'atom_res' in data:
            self.at_resolved = data['atom_res']
            if self.at_resolved and mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + self.sep)
                log.info("\t ATOM RESOLVED CALCULATION")
                log.info("\t " + self.sep)
                log.info("\n")
        #
        #  phonon resolved
        if 'phonon_res' in data:
            self.ph_resolved = data['phonon_res']
            if self.ph_resolved and mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + self.sep)
                log.info("\t PHONON RESOLVED CALCULATION")
                log.info("\t " + self.sep)
                log.info("\n")
            if 'ph_list' in data:
                stph = data['ph_list'][0]
                self.nphr = data['ph_list'][1]
                self.phm_list = list(np.arange(stph, self.nphr+stph))
            if 'nwbn' in data:
                self.nwbn = data['nwbn']
        #
        # 2nd order ZFS / HFI corrections
        if '2nd_order_correct' in data:
            self.order_2_correct = data['2nd_order_correct']
            if 'hessian' in data:
                self.hessian = data['hessian']
        # fraction atoms to be used in the gradient calculation
        # starting from the farthest away from defect
        if 'frac_kept_atoms' in data:
            self.frac_kept_atoms = data['frac_kept_atoms']
        # temperature (K)
        if 'temperature' in data:
            Tlist = data['temperature']
            self.temperatures = set_temperatures(Tlist)
            self.ntmp = len(self.temperatures)
        # eta decay parameter
        if 'eta' in data:
            self.eta = float(data['eta'])
            # eV units
        # min. frequency
        # THz
        if 'min_freq' in data:
            self.min_freq = data['min_freq']
        if mpi.rank == mpi.root:
            log.info("\t min. freq. (THz): " + str(self.min_freq))
            if np.abs(self.min_freq) < 1.E-7:
                log.info("\n")
                log.info("\t " + self.sep)
                log.warning("\t CHECK -> min_freq = " + str(self.min_freq) + " THz")
                log.info("\t " + self.sep)
                log.info("\n")
        # --------------------------------------------------------------
        #
        #    time variables
        #
        # --------------------------------------------------------------
        if 'T' in data:
            self.time_resolved = True
            # time variables
            # ps units
            if len(data['T']) == 2:
                self.T2 = float(data['T'][1])
            # T2 calculation
            # methods
            if 'T2_extract_method' in data:
                if data['T2_extract_method'] == "fit":
                    self.ACF_FIT = True
                    self.ACF_INTEG = False
                elif data['T2_extract_method'] == "integ":
                    self.ACF_FIT = False
                    self.ACF_INTEG = True
                else:
                    log.error("\t T2 EXTRACTION METHOD ONLY : [ fit / integ ]")
                # fitting model -> (1) Exp ; (2) ExpSin
                # ExpSin -> more accurate dynamical calculations
                if self.ACF_FIT and 'fit_model' in data:
                    if data['fit_model'] == "Exp":
                        self.FIT_MODEL = 'Ex'
                    elif data['fit_model'] == "ExpSin":
                        self.FIT_MODEL = 'ExS'
                    else:
                        log.error("\t fit model ONLY : [ Exp / ExpSin ]")
        # --------------------------------------------------------------
        #
        #    frequency variables
        #
        # --------------------------------------------------------------
        if 'nwg' in data:
            self.w_resolved = True
            # n. w grid points
            self.nwg = data['nwg']
            # max. freq. (THz)
            self.w_max = data['w_max']
        #
        # read displ. atoms data
        self.read_atoms_displ()
    #
    # set wql grid -> ph. res.
    #
    def set_wql_grid(self, wu, nq, nat):
        max_freq = np.max(wu)
        # THz
        min_freq = np.inf
        nphm = 3*nat
        for iq in range(nq):
            wuq = wu[iq]
            for iph in range(nphm):
                if wuq[iph] < min_freq and wuq[iph] > self.min_freq:
                    min_freq = wuq[iph]
        max_freq += min_freq / 10.
        dw = (max_freq - min_freq) / self.nwbn
        self.wql_grid = np.zeros(self.nwbn)
        for iwb in range(self.nwbn):
            self.wql_grid[iwb] = min_freq + iwb * dw
        # wql grid index
        self.wql_grid_index = np.zeros((nq, nphm), dtype=int)
        self.wql_freq = np.zeros(self.nwbn, dtype=int)
        for iq in range(nq):
            wuq = wu[iq]
            for iph in range(nphm):
                if wuq[iph] > self.min_freq:
                    ii = int(np.floor((wuq[iph]-min_freq)/dw))
                    self.wql_grid_index[iq,iph] = ii
                    # wql freq.
                    self.wql_freq[ii] += 1
    # read atomic displacements
    def read_atoms_displ(self):
        for i in range(len(self.displ_poscar_dir)):
            file_name = self.displ_poscar_dir[i] + '/displ.yml'
            try:
                f = open(file_name)
            except:
                msg = "\t COULD NOT FIND : " + file_name
                log.error(msg)
            data = yaml.load(f, Loader=yaml.Loader)
            self.atoms_displ.append(np.array(data['displ_ang']))
            f.close()
        # 2nd order
        if self.order_2_correct:
            for i in range(len(self.displ_2nd_poscar_dir)):
                file_name = self.displ_2nd_poscar_dir[i] + '/displ.yml'
                try:
                    f = open(file_name)
                except:
                    msg = "\t COULD NOT FIND : " + file_name
                    log.error(msg)
                data = yaml.load(f, Loader=yaml.Loader)
                self.atoms_2nd_displ.append(np.array(data['displ_ang']))
                f.close()

class linear_resp_input(dynamical_data_input):
    # initialization
    def __init__(self):
        super().__init__()
        # time resolved calculation
        self.time_resolved = False
        # freq. resolved
        self.w_resolved = False
        ####################################
        # freq. resolved calculation inputs
        self.w_grid = None
        # freq. w grid
        self.nwg = 0
        self.w_max = 0.
        # eV units
        self.lorentz_thres = 0.
        # lorentzian treshold
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        self.read_yml_data_dyn(data)
        # only T or nwg in data -> either time or freq. resolved
        if 'T' in data and 'nwg' in data:
            log.error("\t ONLY T / nwg CAN BE IN INPUT DATA -> EITHER TIME OR FREQ. RESOLVED")
        # --------------------------------------------------------------
        #
        #    frequency variables
        #
        # --------------------------------------------------------------
        if 'nwg' in data:
            self.w_resolved = True
            # n. w grid points
            self.nwg = data['nwg']
            # min. freq (THz)
            if 'min_freq' in data:
                self.min_freq = data['min_freq']
            # lorentz. threshold
            if 'lorentz_thres' in data:
                self.lorentz_thres = data['lorentz_thres']
        if mpi.rank == mpi.root:
            if np.abs(self.min_freq) < 1.E-7:
                log.info("\n")
                log.info("\t " + self.sep)
                log.warning("\t CHECK -> min_freq = " + str(self.min_freq) + " THz")
                log.info("\t " + self.sep)
                log.info("\n")
    #
    # set w_grid
    def set_w_grid(self, wu):
        self.w_max = np.max(wu) * THz_to_ev * 10.
        # eV
        dw = self.w_max / (self.nwg - 1)
        self.w_grid = np.zeros(self.nwg)
        # compute w grid
        for iw in range(self.nwg):
            self.w_grid[iw] = iw * dw

class real_time_SQ_input(dynamical_data_input):
    # initialization
    def __init__(self):
        super().__init__()
        #####################################
        # real time parameters
        self.dynamical_mode = []
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        self.read_yml_data_dyn(data)
        # ---------------------------------------------------------------
        #
        #   real time
        #
        # ---------------------------------------------------------------
        if 'dynamics' in data:
            for i in data['dynamics']:
                self.dynamical_mode.append(i)
    #
    # check parameters consistency
    def check_consistency(self):
        # dynamical mode
        assert(len(self.dynamical_mode) == 2)
        if self.order_2_correct:
            if self.dynamical_mode[1] == 0:
                log.warning("\t inconsistent choice of dynamical_mode and order_2_correct: \n")
                log.warning("\t set order_2_correct to False for consistency")
                self.order_2_correct = False
        if self.dynamical_mode[1] > 0:
            if self.order_2_correct == False:
                log.warning("\t inconsistent choice of dynamical_mode and order_2_correct: \n")
                log.warning("\t set order_2_correct to True for consistency")
                self.order_2_correct = True

class real_time_elec_input(ABC):
    def __init__(self):
        self.dynamical_mode = []
        # working directory
        self.work_dir = ''
        # write directory
        self.write_dir = ''
    def read_yml_data_dyn(self, data):
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        if 'output_dir' in data:
            self.write_dir = data['output_dir']
            # create output directory
            if mpi.rank == mpi.root:
                isExist = os.path.isdir(self.write_dir)
                if not isExist:
                    # create new dir.
                    os.makedirs(self.write_dir)
            mpi.comm.Barrier()
        # dynamical mode is organized as follows :
        # len(dynamical_mode) -> 3 : (ee, eph, erad)
        # 0 -> do not include interaction
        # 1 -> do Markovian / Lindblad dyn
        # 2 -> full non Markovian dyn
        if 'dynamics' in data:
            for i in data['dynamics']:
                self.dynamical_mode.append(i)
    def check_consistency(self):
        # dyn. mode
        assert (len(self.dynamical_mode) == 3)
        assert (all(0 <= x <= 2 for x in self.dynamical_mode))

class real_time_JDFTx_input(real_time_elec_input):
    def __init__(self):
        super().__init__()
        #  local input variables
        self.eigenv_file = ''
        self.bnd_kpts_file = ''
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        self.read_yml_data_dyn(data)
        # read specific jdftx files
        if 'bnd_kpts_file' in data:
            self.bnd_kpts_file = self.work_dir + '/' + data['bnd_kpts_file']
        if 'eigenv_file' in data:
            self.eigenv_file = self.work_dir + '/' + data['eigenv_file']
        if 'dft_outfile' in data:
            self.dft_outfile = self.work_dir + '/' + data['dft_outfile']
        if 'wan_cellmap_file' in data:
            self.cellmap_file = self.work_dir + '/' + data['wan_cellmap_file']
        if 'wan_weights_file' in data:
            self.wan_weights_file = self.work_dir + '/' + data['wan_weights_file']
        if 'wan_mlwfh_file' in data:
            self.wan_mlwfh_file = self.work_dir + '/' + data['wan_mlwfh_file']
        if 'ph_outfile' in data:
            self.ph_outfile = self.work_dir + '/' + data['ph_outfile']
        if 'ph_eigenv_file' in data:
            self.ph_eigenv_file = self.work_dir + '/' + data['ph_eigenv_file']
        if 'ph_cellmap_file' in data:
            self.ph_cellmap_file = self.work_dir + '/' + data['ph_cellmap_file']
        if 'ph_basis_file' in data:
            self.ph_basis_file = self.work_dir + '/' + data['ph_basis_file']

class real_time_VASP_input(real_time_elec_input):
    def __init__(self):
        super().__init__()
        #  local input variables
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        self.read_yml_data_dyn(data)

class Q_real_time_input(dynamical_data_input):
    # initialization
    def __init__(self):
        super().__init__()
        # qubutization mode
        self.fermion2qubit = None
        self.qubit_ph = False
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        self.read_yml_data_dyn(data)
        # quantum phonons
        if 'QUANTUM_PHONONS' in data:
            self.qubit_ph = data['QUANTUM_PHONONS']
        # qubitization mode
        if 'fermion2qubit' in data:
            self.fermion2qubit = data['fermion2qubit']
                
class static_data_input(data_input):
    # initialization
    def __init__(self):
        super().__init__()
        # time in musec
        # nuclear spin evolution
        self.T_mus = 0.
        self.dt_mus= 0.
        # dynamical decoupl.
        # parameters
        # n. pulses
        self.n_pulses = 0
        # order taylor exp.
        self.order_exp = 7
    #
    # read yml data file
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #
        # reaad common shared data
        #
        self.read_yml_common_data(data)
        #
        # read nuclear spin dynamics
        # variables
        if 'T_nuc' in data:
            self.T_mus = float(data['T_nuc'])
        if 'dt_nuc' in data:
            self.dt_mus = float(data['dt_nuc'])
        # taylor exp. order
        if 'order_taylor_exp' in data:
            self.order_exp = data['order_taylor_exp']
        # dynamical decoupling -> number of pulses
        if 'npulses' in data:
            self.n_pulses = data['npulses']

class preproc_data_input():
    # initialization
    def __init__(self):
        # working dir.
        self.work_dir = ''
        # unperturbed directory
        self.unpert_dir = ''
        # displ. POSCAR dir
        self.displ_poscar_dir = []
        # displ. OUTCAR dir.
        self.displ_outcar_dir = []
        # copy files dir.
        self.copy_files_dir = ''
        # displ. list
        self.atoms_displ = []
        # max d_ab
        self.max_dab = np.inf
        # max. dist from defect
        self.max_dist_defect = np.inf
        # defect index
        self.defect_index = None
    #
    # read yml data file
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #
        #    extract directories
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        # unpert. data directory
        if 'unpert_dir' in data:
            self.unpert_dir = self.work_dir + '/' + data['unpert_dir']
        # displ. POSCAR dir.
        if 'displ_poscar_dir' in data:
            for d in data['displ_poscar_dir']:
                self.displ_poscar_dir.append(self.work_dir + '/' + d)
        # displ. OUTCAR dir.
        if 'displ_outcar_dir' in data:
            for d in data['displ_outcar_dir']:
                self.displ_outcar_dir.append(self.work_dir + '/' + d)
        # copy files dir.
        if 'copy_files_dir' in data:
            self.copy_files_dir = self.work_dir + '/' + data['copy_files_dir']
        # displ. ang.
        if 'displ_ang' in data:
            for d in data['displ_ang']:
                self.atoms_displ.append(np.array(d))
        # max d(a,b)
        if 'max_dab' in data:
            self.max_dab = data['max_dab']
        if 'max_dist_from_defect' in data:
            self.max_dist_defect = data['max_dist_from_defect']
        # index defect
        if 'defect_index' in data:
            self.defect_index = data['defect_index']
