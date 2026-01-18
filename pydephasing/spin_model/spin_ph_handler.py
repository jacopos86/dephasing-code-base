from pydephasing.spin_model.spin_ph_inter import SpinPhononFirstOrder
from pydephasing.spin_model.spin_ph_inter_secnd_order import SpinPhononSecndOrder

#
#    function to handle the
#    spin phonon interaction class
#

def spin_ph_handler(secnd_order, ZFS_CALC, HFI_CALC, HESSIAN):
     # if not second order
     # define first order spin - phonon
     if not secnd_order:
          return SpinPhononFirstOrder(ZFS_CALC, HFI_CALC)
     else:
          return SpinPhononSecndOrder().generate_instance(ZFS_CALC, HFI_CALC, HESSIAN)