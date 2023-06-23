#
from class_hfi_zfs import SpinTriplet
#
st = SpinTriplet("./examples/NV-diamond/GS")
st.read_poscar()
st.read_zfs_tensor()
st.read_hfi_full()
st.set_hfi_Dbasis()
# HFI in MHz st.Ahfi[ia,ix,iy] FC + DIPOLAR
ia = 110
print(st.Ahfi[ia-1,:,:])
# ZFS in MHz (diagonal)
print(st.Ddiag)
