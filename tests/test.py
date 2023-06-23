#
#  test script
#
from pydephasing.set_structs import DisplacedStructs, UnpertStruct
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.gradient_interactions import gradient_ZFS, gradient_HFI, gradient_Eg
import numpy as np
import sys
import scipy.linalg as la
#
atoms_unprt = UnpertStruct("./examples/NV-diamond/GS")
atoms_unprt.read_poscar()
atoms_unprt.read_zfs_tensor()
U = atoms_unprt.Deigv
DU = np.matmul(atoms_unprt.Dtensor, U)
D = np.matmul(U.transpose(), DU)
# struct. displ. A
struct_displ_A = DisplacedStructs("./examples/NV-diamond/DISPLACEMENT-FILES-0001", "./examples/NV-diamond/DISPL-0001")
struct_displ_A.atom_displ(np.array([0.001, 0.001, 0.001]))    # Ang
struct_displ_A.build_atom_displ_structs(atoms_unprt)
struct_displ_A.write_structs_on_file()
# struct. displ. B
struct_displ_B = DisplacedStructs("./examples/NV-diamond/DISPLACEMENT-FILES-01", "./examples/NV-diamond/DISPL-01")
struct_displ_B.atom_displ(np.array([0.1, 0.1, 0.1]))          # Ang
struct_displ_B.build_atom_displ_structs(atoms_unprt)
struct_displ_B.write_structs_on_file()
# grad. interactions
structs_pert = [struct_displ_A, struct_displ_B]
gradHFI = gradient_HFI(atoms_unprt.nat)
gradHFI.set_grad_Ahfi(struct_displ_A, atoms_unprt.nat)
gradHFI.set_grad_Ahfi_Ddiag_basis(atoms_unprt)
gradZFS = gradient_ZFS(atoms_unprt.nat)
gradZFS.set_tensor_gradient(structs_pert, atoms_unprt, "./examples/NV-diamond/info", "./examples/NV-diamond/")
gradZFS.set_grad_D_tensor(atoms_unprt)
#gradZFS.plot_tensor_grad_component(0, 1, structs_pert, atoms_unprt, "./examples/NV-diamond/")
#gradEg = gradient_Eg(atoms_unprt.nat)
#gradEg.set_gradEg(displ_structs, nat, "./examples/GCB/GS/DISPL/", "./examples/GCB/EXC/DISPL/")
#gradZFS.set_grad_D_tensor(atoms_unprt)
#
sys.exit()
Hss = spin_hamiltonian()
spin_config = nuclear_spins_config(10)
spin_config.set_time(0.01, 300.)
spin_config.set_nuclear_spins(atoms_unprt.nat)
spin_config.set_nuclear_spin_evol(np.array([0.,0.,100.]), Hss, atoms_unprt)
sys.exit()
#
qs1 = np.array([1.,0.,0.])
qs2 = np.array([0.,1.,0.])
lambda_coef = np.array([1.,0.])
#Hss.set_grad_Ess(gradZFS, gradHFI, spin_config, qs, atoms_unprt.nat, lambda_coef)
Hss.set_grad_deltaEss(gradZFS, gradHFI, spin_config, qs1, qs2, atoms_unprt.nat, lambda_coef)
#Hss.diagonalize_Hss(atoms_unprt.Ddiag)
sys.exit()
#
E_S1 = Hss.set_E_coef(qs)
E_A1 = Hss.set_A_coef(qs)
#qs = np.array([1.,-1j,0.])/np.sqrt(2.)
qs = np.array([0.,1.,0.])
E_S2 = Hss.set_E_coef(qs)
E_A2 = Hss.set_A_coef(qs)
print(E_A1-E_A2)
qs1 = qs
qs2 = np.array([0.,1.,0.])
Hss.set_grad_deltaEss(gradZFS, gradHFI, spin_config, qs1, qs2, atoms_unprt.nat, lambda_coef)
