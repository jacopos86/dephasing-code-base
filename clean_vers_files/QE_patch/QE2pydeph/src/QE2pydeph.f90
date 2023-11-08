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
  USE io_global,                 ONLY : ionode, ionode_id, stdout
  USE mp_pools,                  ONLY : npool
  USE mp,                        ONLY : mp_bcast
  USE environment,               ONLY : environment_start, environment_end
  USE input_parameters,          ONLY : ZFS, HFI, nconfig
  USE zfs_module,                ONLY : compute_ddig_space, compute_invfft_ddiG,     &
       allocate_array_variables, compute_zfs_tensor, set_spin_band_index_occ_levels
  USE noncollin_module,          ONLY : npol
  
  
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
  
  NAMELIST / inputQE_pydeph / outdir, prefix, ZFS, HFI, nconfig
  
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
  call mp_bcast (nconfig, ionode_id, intra_image_comm)

  !
  !  allocate space for pwscf variables
  !

  call read_file
  call openfil_pp

  call weights ( )
  !
  call init_us_1
  
  !
  !  prepare ZFS calculation : if required
  !
  
  IF (ZFS) THEN
     !
     IF (npol > 1) call errore (code, 'non collinearity not implemented', npol)
     
     !
     !  set nmax and arrays
     
     call set_spin_band_index_occ_levels ( )
     
     !
     
     call allocate_array_variables ( )
     
     !
     !  compute ddi (G)
     !
     
     call compute_ddig_space ( )
     
     !
     !  compute ddi real space
     !
     
     call compute_invfft_ddiG ( )
     
     !
     !  compute Dab
     !
     
     call compute_zfs_tensor ( )
     
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
