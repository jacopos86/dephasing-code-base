import numpy as np
from abc import ABC, abstractmethod

#  This modules computes 
#  different observables matrix elements
#  and their expectation values
#  from the system density matrix

class Observables(ABC):
    def __init__(self, basis_set):
        self.basis_set = basis_set
        self.Sx = None
        self.Sy = None
        self.Sz = None
    @abstractmethod
    def set_Spin_operators(self, *args, **kwargs):
        pass
    @abstractmethod
    def compute_spin_mag(self, dm_obj):
        pass
    @abstractmethod
    def compute_occup_prob(self, dm_obj):
        pass

class ObservablesSpinModel(Observables):
    def __init__(self, basis_set):
        super().__init__(basis_set)
    def set_Spin_operators(self, H):
        pass
    #
    #    compute spin magnetization
    #    expect. value
    def compute_spin_mag(self, dm_obj):
        # compute magnet. expect. value
        rho = dm_obj.matr
        nt = rho.shape[-1]
        M_oft = np.zeros((3, nt))
        # run on time array
        for t in range(nt):
            M_oft[0,t] = np.matmul(rho[:,:,t], self.Sx).trace().real
            M_oft[1,t] = np.matmul(rho[:,:,t], self.Sy).trace().real
            M_oft[2,t] = np.matmul(rho[:,:,t], self.Sz).trace().real
        return M_oft
    #  
    #   compute occupations probability
    def compute_occup_prob(self, dm_obj):
        # compute occup. prob.
        rho = dm_obj.rho
        nt = rho.shape[-1]
        n = rho.shape[0]
        occup = np.zeros((n, nt))
        # run over time array
        for t in range(nt):
            for i in range(n):
                occup[i,t] = (rho[i,i,t] * rho[i,i,t].conj()).real
        return occup
    
class ObservablesElectronicSystem(Observables):
    def __init__(self, basis_set):
        super().__init__(basis_set)
    #
    #   define spin matrix elements
    def set_Spin_operators(self, struct_0):
        self.Sx = np.zeros((struct_0.nbnd, struct_0.nbnd, struct_0.nkpt), dtype=np.complex128)
        self.Sy = np.zeros((struct_0.nbnd, struct_0.nbnd, struct_0.nkpt), dtype=np.complex128)
        self.Sz = np.zeros((struct_0.nbnd, struct_0.nbnd, struct_0.nkpt), dtype=np.complex128)
        if struct_0.spinor_wfc:
            isp = 1
            for ik in range(struct_0.nkpt):
                npw = self.basis_set.get_pw_number(ik+1)
                ng = npw // 2
                for ib1 in range(struct_0.nbnd):
                    cnk1 = self.basis_set.read_cnk_ofG(isp, ik+1, ib1+1, norm=True)
                    cnk1_up = cnk1[:ng]
                    cnk1_dw = cnk1[ng:]
                    cnk1 = np.column_stack((cnk1_up, cnk1_dw))
                    for ib2 in range(ib1, struct_0.nbnd):
                        cnk2 = self.basis_set.read_cnk_ofG(isp, ik+1, ib2+1, norm=True)
                        cnk2_up = cnk2[:ng]
                        cnk2_dw = cnk2[ng:]
                        cnk2 = np.column_stack((cnk2_up, cnk2_dw))
                        self.Sz[ib1,ib2,ik] = np.vdot(cnk1[:,0], cnk2[:,0]) - np.vdot(cnk1[:,1], cnk2[:,1])
                        self.Sx[ib1,ib2,ik] = np.vdot(cnk1[:,0], cnk2[:,1]) + np.vdot(cnk1[:,1], cnk2[:,0])
                        self.Sy[ib1,ib2,ik] = -1j*np.vdot(cnk1[:,0], cnk2[:,1]) + 1j*np.vdot(cnk1[:,1], cnk2[:,0])
                        if ib2 > ib1:
                            self.Sz[ib2,ib1,ik] = np.conj(self.Sz[ib1,ib2,ik])
                            self.Sx[ib2,ib1,ik] = np.conj(self.Sx[ib1,ib2,ik])
                            self.Sy[ib2,ib1,ik] = np.conj(self.Sy[ib1,ib2,ik])
                for ib in range(struct_0.nbnd):
                    print(ib, self.Sz[ib,ib,0], struct_0.occ[ib,0])
    def compute_spin_mag(self, dm_obj):
        pass
    def compute_occup_prob(self, dm_obj):
        pass