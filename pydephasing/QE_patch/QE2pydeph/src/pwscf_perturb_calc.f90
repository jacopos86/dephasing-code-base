!
!   Author : Jacopo Simoni
!   This file contains the subroutines needed
!   to compute the GS electronic structure
!   for each perturbed atomic system
!   it uses run_pwscf.f90 in PW/src/
!   without the atom dynamics part
!
! ===========================================================================
SUBROUTINE run_pwscf_perturb_atom_struct ( exit_status )
  ! =========================================================================
  !! Run an instance of the Plane Wave Self-Consistent Field code 
  !! MPI initialization and input data reading is performed in the 
  !! calling code - returns in exit_status the exit code for pw.x, 
  !! returned in the shell. Values are:  
  !! * 0: completed successfully
  !! * 1: an error has occurred (value returned by the errore() routine)
  !! * 2-127: convergence error
  !!    * 2: scf convergence error
  !!    * 3: ion convergence error
  !! * 128-255: code exited due to specific trigger
  !!    * 255: exit due to user request, or signal trapped,
  !!          or time > max_seconds
  !!     (note: in the future, check_stop_now could also return a value
  !!     to specify the reason of exiting, and the value could be used
  !!     to return a different value for different reasons)








  !
  implicit none













  !
  exit_status = 0
  IF (ionode) WRITE( UNIT=stdout, FMT=9010 ) ntypx, npk, lmaxx
  !
  IF (ionode) call plugin_arguments ()
  call plugin_arguments_bcast( ionode_id, intra_image_comm )
  !
  ! ... needs to come before iosys() so some input flags can be
  !     overridden without needing to write PWscf specific code.
  ! 
  call qmmm_initialization()
  !
  ! ... convert to internal variables
  !
  call iosys()
  !
  IF ( gamma_only ) WRITE ( UNIT = stdout,   &
       & FMT = '(/,5X,"gamma-point specific algorithms are used")' )
  !
  ! call to void routine for user defined / plugin patches initializations
  !
  call plugin_initialization()
  !
  call check_stop_init()
  !
  call setup()
  !
  call qmmm_update_positions()
  !
  ! ... dry run: code will stop here if called with exit file present
  ! ... useful for a quick and automated way to check input data
  !
  IF ( nconfig == 0 .OR. check_stop_now() ) THEN
     CALL pre_init()
     CALL data_structure( gamma_only )
     CALL summary()
     CALL memory_report()
     CALL qexsd_set_status(255)
     CALL punch( 'config-init' )
     exit_status = 255
     RETURN
  ENDIF
  !
  call init_run ()
  !
  IF ( check_stop_now() ) THEN
     CALL qexsd_set_status( 255 )
     CALL punch( 'config' )
     exit_status = 255
     RETURN
  ENDIF
  !
  !  main loop over atomic configurations
  !
  main_loop: DO ic= 1, nconfig
     !
     !   electronic self consistency
     !   calculation
     !
     










     !
     !   here :
     !   compute different quantities
     !   save results on files
     !




     
     !
  ENDDO main_loop

  !
9010 FORMAT( /,5X,'Current dimensions of program PWSCF are:', &
           & /,5X,'Max number of different atomic species (ntypx) = ',I2,&
           & /,5X,'Max number of k-points (npk) = ',I6,       &
           & /,5X,'Max angular momentum in pseudopotentials (lmaxx) = ',i2)
9020 FORMAT( /,5X,'Final scf calculation at the relaxed structure.', &
          &  /,5X,'The G-vectors are recalculated for the final unit cell', &
          &  /,5X,'Results may differ from those at the preceding step.' )
  !
END SUBROUTINE run_pwscf_perturb_atom_struct
