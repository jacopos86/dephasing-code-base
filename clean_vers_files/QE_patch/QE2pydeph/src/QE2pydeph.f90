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
  USE input_parameters,          ONLY : ZFS, HFI, nconfig, SOC_CORR
  USE zfs_module,                ONLY : compute_ddig_space, compute_invfft_ddiG,     &
       allocate_array_variables, compute_zfs_tensor, set_spin_band_index_occ_levels, &
       set_SOC_transitions_list, compute_soc_zfs_tensor
  USE noncollin_module,          ONLY : npol
  USE spin_orbit_operator,       ONLY : read_FR_pseudo, frpsfile, read_FR_pseudo_from_file,  &
       set_spin_orbit_operator, init_run_frpp, dvan_so_pauli_basis, compute_soc_matrix_elements
  USE funct,                     ONLY : get_dft_name
  USE spin_orb,                  ONLY : lspinorb
  USE klist,                     ONLY : nks
  USE bec_module,                ONLY : allocate_bec_arrays, compute_bec_array
  USE wvfct,                     ONLY : nbnd
  
  !
  IMPLICIT NONE
  
  !
  !   INTERNAL VARIABLES
  !
  
  CHARACTER (LEN=256)              :: outdir
  CHARACTER (LEN=8)                :: code = 'QE_PYDEPH'
  CHARACTER (LEN=256)              :: trimcheck
  CHARACTER (LEN=20)               :: dft_name
  !
  INTEGER                          :: ios
  integer                          :: ntr
  integer, allocatable             :: transitions_list (:,:)
  
  !
  !   NAMELIST zfs_hfi module
  
  NAMELIST / inputQE_pydeph / outdir, prefix, ZFS, HFI, nconfig, SOC_CORR
  
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
  call mp_bcast (SOC_CORR, ionode_id, intra_image_comm)
  
  !
  !  allocate space for pwscf variables
  !
  
  call read_file
  call openfil_pp
  
  call weights ( )
  !
  IF (.not. SOC_CORR) call init_us_1
  
  IF (lspinorb) call errore (code, 'lspinorb must be .false.', 1)
  
  !
  IF (npol > 1) call errore (code, 'non collinearity not implemented', npol)
  
  WRITE(stdout,*) "    nks= ", nks, nbnd
  
  !
  IF (ionode) THEN
     !
     !  ... READ frpp IF NEEDED
     !
     
     IF (SOC_CORR) call read_FR_pseudo_from_file ()
     !
  END IF
  
  IF (SOC_CORR) call mp_bcast (frpsfile, ionode_id, intra_image_comm)
  
  !
  !  prepare ZFS calculation : if required
  !
  
  IF (ZFS) THEN
     
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
     !  spin orbit section
     !
     WRITE(stdout,*) "SOC_CORR: ", SOC_CORR
     
     !
     IF (SOC_CORR) THEN
        !
        
        dft_name = get_dft_name ()
        call read_FR_pseudo ( dft_name, .TRUE. )
        !
        !  set SOC operator
        call init_run_frpp ()
        !
        call set_spin_orbit_operator ()
        !
        call dvan_so_pauli_basis ()
        
        !
        !  compute <beta|psi>
        !
        call allocate_bec_arrays ()
        !
        call compute_bec_array ()
        !
        call set_SOC_transitions_list ( transitions_list, ntr )
        !
        call compute_soc_matrix_elements ( transitions_list, ntr )
        !
        call compute_soc_zfs_tensor ( transitions_list, ntr )
        !
     END IF
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
