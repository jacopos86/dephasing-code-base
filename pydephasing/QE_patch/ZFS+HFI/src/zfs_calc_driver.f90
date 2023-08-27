  !
  ! ----------------------------------------------------------------------------
  !
  !    This program reads an input file and produce in output
  !    a file with the ZFS tensor
  !
  ! ----------------------------------------------------------------------------
PROGRAM zfs_calculation
  ! ----------------------------------------------------------------------------
  !
  
  USE io_files,                  ONLY : prefix
  USE mp_global,                 ONLY : mp_startup
  
  
  
  IMPLICIT NONE
  
  !
  !   INTERNAL VARIABLES
  !
  
  CHARACTER (LEN=256)              :: outdir
  
  
  
  
  
  !
  !   NAMELIST zfs_hfi module
  
  NAMELIST / inputzfs_hfi / outdir, prefix
  
#ifdef __MPI
  call mp_startup ()
#endif
  
  
  
  
  
  
  










  
  !
  STOP
  
  !
END PROGRAM zfs_calculation
