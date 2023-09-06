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
  
  USE io_files,                  ONLY : prefix, tmp_dir
  USE mp_global,                 ONLY : mp_startup
  USE mp_images,                 ONLY : intra_image_comm
  USE io_global,                 ONLY : ionode, ionode_id
  USE mp_pools,                  ONLY : npool
  USE mp,                        ONLY : mp_bcast
  USE environment,               ONLY : environment_start, environment_end
  
  
  
  IMPLICIT NONE
  
  !
  !   INTERNAL VARIABLES
  !
  
  CHARACTER (LEN=256)              :: outdir
  CHARACTER (LEN=8)                :: code = 'ZFS_CALC'
  CHARACTER (LEN=256)              :: trimcheck
  !
  INTEGER                          :: ios
  
  
  
  
  !
  !   NAMELIST zfs_hfi module
  
  NAMELIST / inputzfs_hfi / outdir, prefix
  
#ifdef __MPI
  call mp_startup ()
#endif

  !
  call environment_start (code)

  !
  prefix = 'zfs'
  call get_environment_variable ('ESPRESSO_TMPDIR', outdir)
  IF ( TRIM (outdir) == ' ' ) outdir = './'

  !
  IF (npool > 1) call errore (code, 'pools not implemented', npool)
  !
  IF (ionode) THEN
     !
     call input_from_file ()
     !
     READ (5, inputzfs_hfi, err=200, iostat=ios)
200  call errore (code, 'reading inputzfs_hfi namelist', abs (ios))

     !
     tmp_dir = trimcheck (outdir)
     !
  END IF

  !
  !  ... bcast
  !

  call mp_bcast (tmp_dir, ionode_id, intra_image_comm)
  call mp_bcast (prefix, ionode_id, intra_image_comm)
  
  !
  IF (npool > 1) call errore (code, 'pools not implemented', npool)
  
  
  
  
  









  call environment_end (code)
  !
  
  call stop_pp
  STOP
  
  !
END PROGRAM zfs_calculation
