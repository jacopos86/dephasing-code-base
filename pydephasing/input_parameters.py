#
#   This module defines all the
#   parameters needed in input for the
#   calculations
#
import os
import numpy as np
import yaml
from pydephasing.log import log
from pydephasing.mpi import mpi
#
class data_input():
    # initialization
    def __init__(self):
        # working directory
        self.work_dir = ''
        # write directory
        self.write_dir = ''
        # unperturbed directory
        self.unpert_dirs = []
        # perturbed calculations directory
        self.displ_poscar_dir = []
        self.displ_2nd_poscar_dir = []
        # perturbed calculations outcars directory
        self.displ_outcar_dir = []
        self.displ_2nd_outcar_dir = []
        # copy directory
        self.copy_files_dir = ''
        # grad info file
        self.grad_info = ''
        # atoms displacements
        self.atoms_displ = []
        self.atoms_2nd_displ = []
        # at resolved
        self.at_resolved = False
        # ph. resolved
        self.ph_resolved = False
        # relax. type calculation
        self.relax = False
        # dephasing type calculation
        self.deph = False
        # dynamical decoupling F by default
        self.dyndec = False
        # n. temperatures
        self.ntmp = 0
        # FC core
        self.fc_core = True
        # nlags
        self.nlags = 0
        # time
        self.T  = 0.0
        self.T2 = 0.0
        self.T_mus = 0.0
        # eta decay parameter
        self.eta = 1e-3
        # dephasing func. param.
        self.N_df= 100000
        self.T_df= 0.05
        # in ps units
        self.maxiter = 80000
        # n. spins list
        self.nsp = 0
        # zfs 2nd order correction
        self.order_2_correct = False
        # min. freq.
        self.min_freq = 0.0                    
        # THz
        self.n_pulses = None
        # dyn dec n. pulses
        self.nk = 7
        # order derivative expansion
    #
    # read yaml data
    def read_yml_data(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # extract directories
        # path always with respect to working directory
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        if 'displ_poscar_dir' in data:
            for d in data['displ_poscar_dir']:
                self.displ_poscar_dir.append(self.work_dir + '/' + d)
        if 'displ_2nd_poscar_dir' in data:
            for d in data['displ_2nd_poscar_dir']:
                self.displ_2nd_poscar_dir.append(self.work_dir + '/' + d)
        if 'displ_outcar_dir' in data:
            for d in data['displ_outcar_dir']:
                self.displ_outcar_dir.append(self.work_dir + '/' + d)
        if 'displ_2nd_outcar_dir' in data:
            for d in data['displ_2nd_outcar_dir']:
                self.displ_2nd_outcar_dir.append(self.work_dir + '/' + d)
        if 'output_dir' in data:
            self.write_dir = data['output_dir']
            self.write_dir = self.work_dir + '/' + self.write_dir
            # create output directory
            if mpi.rank == mpi.root:
                isExist = os.path.isdir(self.write_dir)
                if not isExist:
                    # create new dir.
                    os.makedirs(self.write_dir)
            mpi.comm.Barrier()
        if 'grad_info_file' in data:
            self.grad_info = self.work_dir + '/' + data['grad_info_file']
        if 'yaml_pos_file' in data:
            self.yaml_pos_file = self.work_dir + '/' + data['yaml_pos_file']
        if 'hd5_eigen_file' in data:
            self.hd5_eigen_file = self.work_dir + '/' + data['hd5_eigen_file']
        if 'atom_res' in data:
            self.at_resolved = data['atom_res']
            if self.dyndec:
                self.at_resolved = False
            if self.at_resolved and mpi.rank == mpi.root:
                log.info("atom resolved calculation")
        if 'phonon_res' in data:
            self.ph_resolved = data['phonon_res']
            if self.dyndec:
                self.ph_resolved = False
            if self.ph_resolved and mpi.rank == mpi.root:
                log.info("phonon resolved calculation")
            if 'ph_list' in data:
                stph = data['ph_list'][0]
                self.nphr = data['ph_list'][1]
                self.phm_list = list(np.arange(stph, self.nphr+stph))
            if 'nwbn' in data:
                self.nwbn = data['nwbn']
        # time variables
        if 'T' in data:
            # in ps units
            if len(data['T']) == 1:
                self.T = float(data['T'][0])
            if len(data['T']) == 2:
                self.T = float(data['T'][0])
                self.T2= float(data['T'][1])
            else:
                log.error("only two T values 1) normal calc. 2) ph/at resolved")
            # min. frequency
            self.min_freq = 1./self.T
            if mpi.rank == mpi.root:
                log.info("min. freq. (THz): " + str(self.min_freq))
        if 'dt' in data:
            # ps
            self.dt = float(data['dt'])
        if 'eta' in data:
            self.eta = float(data['eta'])
        # temperature (K)
        if 'temperature' in data:
            Tlist = data['temperature']
        # dynamical decoupling -> number of pulses
        if 'npulses' in data:
            self.n_pulses = data['npulses']
        # deph. function parameters
        if 'maxiter' in data:
            self.maxiter = int(data['maxiter'])
        if 'Ndf' in data:
            self.N_df = int(data['Ndf'])
        if 'Tdf' in data:
            self.T_df = float(data['Tdf'])
        # HFI calculation parameters
        if 'nconfig' in data:
            self.nconf = int(data['nconfig'])
        if 'nspins' in data:
            self.nsp = int(data['nspins'])
        if 'B0' in data:
            self.B0 = np.array(data['B0'])
            # Gauss units
        # fermi contact term
        if 'core' in data:
            if data['core'] == False:
                self.fc_core = False
        # 2nd order ZFS correction
        if '2nd_order_correct' in data:
            self.order_2_correct = data['2nd_order_correct']
        # set temperature list
        self.ntmp = len(Tlist)
        self.temperatures = np.array(Tlist)
        # set time arrays
        self.nt = int(self.T / self.dt)
        self.time = np.linspace(0., self.T, self.nt)
        # time 2
        self.nt2 = int(self.T2 / self.dt)
        self.time2 = np.linspace(0., self.T2, self.nt2)
        # read displacement data
        for i in range(len(self.displ_poscar_dir)):
            file_name = self.displ_poscar_dir[i] + '/displ.yml'
            try:
                f = open(file_name)
            except:
                msg = "could not find: " + file_name
                log.error(msg)
            data = yaml.load(f, Loader=yaml.Loader)
            self.atoms_displ.append(np.array(data['displ_ang']))
            f.close()
        if self.order_2_correct:
            for i in range(len(self.displ_2nd_poscar_dir)):
                file_name = self.displ_2nd_poscar_dir[i] + '/displ.yml'
                try:
                    f = open(file_name)
                except:
                    msg = "could not find: " + file_name
                    log.error(msg)
                data = yaml.load(f, Loader=yaml.Loader)
                self.atoms_2nd_displ.append(np.array(data['displ_ang']))
                f.close()
    #
    # read inhomo stat. calculation
    #
    def read_inhomo_stat(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # extract directory
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        if 'output_dir' in data:
            self.write_dir = data['output_dir']
            self.write_dir = self.work_dir + '/' + self.write_dir
            # create output directory
            if mpi.rank == mpi.root:
                isExist = os.path.isdir(self.write_dir)
                if not isExist:
                    # create new dir
                    os.makedirs(self.write_dir)
            mpi.comm.Barrier()
        if 'grad_info_file' in data:
            self.grad_info = self.work_dir + '/' + data['grad_info_file']
        if 'yaml_pos_file' in data:
            self.yaml_pos_file = self.work_dir + '/' + data['yaml_pos_file']
        #
        # time variables
        if 'T' in data:
            # ps units
            self.T = float(data['T'])
        if 'dt' in data:
            # ps units
            self.dt = float(data['dt'])
        # spin Ii dynamics
        if 'T_nuc' in data:
            self.T_mus = float(data['T_nuc'])
        if 'dt_nuc' in data:
            self.dt_mus = float(data['dt_nuc'])
        # deph. function params.
        if 'maxiter' in data:
            self.maxiter = int(data['maxiter'])
        if 'Ndf' in data:
            self.N_df = int(data['Ndf'])
        if 'Tdf' in data:
            self.T_df = float(data['Tdf'])
        if 'nlags' in data:
            self.nlags = int(data['nlags'])
        # HFI calculation
        if 'nconfig' in data:
            self.nconf = int(data['nconfig'])
        if 'nspins' in data:
            self.nsp = int(data['nspins'])
        if 'B0' in data:
            self.B0 = np.array(data['B0'])
            # Gauss units
        # fermi contact term
        if 'core' in data:
            if data['core'] == False:
                self.fc_core = False
        # psi0
        self.psi0 = np.zeros(3, dtype=np.complex128)
        if 'psi0' in data:
            self.psi0 = np.array(data['psi0'], dtype=np.complex128)
        else:
            self.psi0 = np.array([1.+0*1j,1.+0*1j,0.+0*1j])
        nrm = np.sqrt(sum(self.psi0[:] * np.conjugate(self.psi0[:])))
        self.psi0 = self.psi0 / nrm
        # set time arrays
        self.nt = int(self.T / self.dt)
        self.time = np.linspace(0., self.T, self.nt)
        if self.nlags == 0:
            self.nlags = self.nt
        # time2
        self.nt2 = int(self.T_mus / self.dt_mus)
        self.time2 = np.linspace(0., self.T2, self.nt2)
    #
    # read pre-processing input
    #
    def read_yml_data_pre(self, input_file):
        try:
            f = open(input_file)
        except:
            msg = "could not find: " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        f.close()
        # extract directories
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        if 'unpert_dir' in data:
            if len(data['unpert_dir']) == 1:
                self.unpert_dir = self.work_dir + '/' + data['unpert_dir'][0]
            else:
                log.error("only one unperturbed data every time")
        if 'displ_poscar_dir' in data:
            for d in data['displ_poscar_dir']:
                self.displ_poscar_dir.append(self.work_dir + '/' + d)
        if 'displ_outcar_dir' in data:
            for d in data['displ_outcar_dir']:
                self.displ_outcar_dir.append(self.work_dir + '/' + d)
        if 'copy_files_dir' in data:
            self.copy_files_dir = self.work_dir + '/' + data['copy_files_dir']
        if 'displ_ang' in data:
            for d in data['displ_ang']:
                self.atoms_displ.append(np.array(d))
        if 'max_dab' in data:
            self.max_dab = data['max_dab']
        if 'max_dist_from_defect' in data:
            self.max_dist_defect = data['max_dist_from_defect']
        if 'defect_index' in data:
            self.defect_index = data['defect_index']
    #
    # set dyn-dec parameters
    #
    def set_dyndec_param(self, wu):
        # compute minimal time interval
        # dtm = 1 / max_freq / 2
        max_freq = np.max(wu)
        # THz
        dtm = 1./max_freq/2
        # ps
        self.nt0 = int(self.time[-1] / dtm)
        it_seq = np.array_split(range(self.nt), self.nt0)
        self.it0_seq = [None]*self.nt0
        self.it1_seq = [None]*self.nt0
        # build time sequences
        for it0 in range(self.nt0):
            x = it_seq[it0]
            self.it0_seq[it0] = x[0]
            self.it1_seq[it0] = x[-1]+1
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
# input parameters object
p = data_input()