!
! ----------------------------------------------------------------------------
!
!    This program is the main driver of the QE - pydephasing
!    interface
!
! ----------------------------------------------------------------------------
PROGRAM QE_pydeph
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
  CHARACTER (LEN=8)                :: code = 'QE_PYDEPH'
  CHARACTER (LEN=256)              :: trimcheck
  !
  INTEGER                          :: ios
  
  
  
  
  !
  !   NAMELIST zfs_hfi module
  
  NAMELIST / inputQE_pydeph / outdir, prefix, ZFS, HFI
  
#ifdef __MPI
  call mp_startup ()
#endif

  !
  call environment_start (code)

  !
  prefix = 'pydeph'
  call get_environment_variable ('ESPRESSO_TMPDIR', outdir)
  IF ( TRIM (outdir) == ' ' ) outdir = './'

  !
  IF (npool > 1) call errore (code, 'pools not implemented', npool)
  !
  IF (ionode) THEN
     !
     call input_from_file ()
     !
     READ (5, inputQE_pydeph, err=200, iostat=ios)
200  call errore (code, 'reading inputQE_pydeph namelist', abs (ios))

     !
     tmp_dir = trimcheck (outdir)
     !
  END IF

  !
  !  ... bcast
  !

  call mp_bcast (tmp_dir, ionode_id, intra_image_comm)
  call mp_bcast (prefix, ionode_id, intra_image_comm)
  call mp_bcast (ZFS, ionode_id, intra_image_comm)
  call mp_bcast (HFI, ionode_id, intra_image_comm)
  
  !
  IF (npool > 1) call errore (code, 'pools not implemented', npool)
  
  
  !
  !  prepare ZFS calculation : if required
  !
  
  IF (ZFS) THEN

     !
     !  compute ddi (G)
     !

     call compute_ddig_space ( )

     !
     !  compute ddi real space
     !
     
     call compute_invfft_ddiG ( )
     
     !
  END IF
  !
  
  
  







  
  !
  call environment_end (code)
  !
  
  call stop_pp
  STOP
  
  !
END PROGRAM QE_pydeph
