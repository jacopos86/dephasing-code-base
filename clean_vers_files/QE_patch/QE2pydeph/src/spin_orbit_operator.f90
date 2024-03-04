!
!   MODULE  :  spin orbit operator
!
MODULE spin_orbit_operator
  
  USE kinds,                        ONLY : DP
  USE pseudo_types,                 ONLY : pseudo_upf
  USE radial_grids,                 ONLY : radial_grid_type
  USE parameters,                   ONLY : ntypx
  !
  
  TYPE (pseudo_upf), ALLOCATABLE, TARGET          :: frpp (:)
  !
  !  fully relativistic pp MUST include SOC
  type(radial_grid_type), allocatable, target     :: rgrid (:)
  !
  integer, allocatable                            :: msh (:)
  !
  LOGICAL                                         :: taspc = .false.
  !
  CHARACTER(len=80)                               :: frpsfile(ntypx) = 'YY'
  !
  !  full relativistic PP files list
  complex(DP), allocatable                        :: Dso (:,:,:,:)
  !
  !  SOC operator
  complex(DP)                                     :: sigma_x (2,2), sigma_y (2,2), sigma_z (2,2)
  !
  !  spin operators
  complex(DP), allocatable                        :: HSO_a (:,:)
  !  spin orbit matrix elements
  
  !
CONTAINS
  !
  ! ====================================================================================
  subroutine set_spin_operators ()
    ! ----------------------------------------------------------------------------------
    
    IMPLICIT NONE
    
    !
    !    sigma operators
    
    sigma_x (1,2) = cmplx (1._dp,0._dp)
    sigma_x (2,1) = cmplx (1._dp,0._dp)
    
    !
    sigma_y (1,2) = cmplx (0._dp, -1._dp)
    sigma_y (2,1) = cmplx (0._dp,  1._dp)
    
    !
    sigma_z (1,1) = cmplx (1._dp, 0._dp)
    sigma_z (2,2) = cmplx (-1._dp, 0._dp)
    
    !
  end subroutine set_spin_operators
  !
  ! ====================================================================================
  subroutine init_run_frpp ( )
    ! ==================================================================================
    
    implicit none
    
    !
    call init_frpp_parameters ( )
    
    !
    call allocate_FR_pp_variables ( )
    
    !
  end subroutine init_run_frpp
  !
  ! ====================================================================================
  subroutine compute_spin_orbit_operator ( )
    ! ----------------------------------------------------------------------------------
    
    implicit none
    
    !
    call set_spin_operators ()
    !
    call set_spin_orbit_operator ()
    !
    call dvan_so_pauli_basis ()
    
    !
    RETURN
    !
  end subroutine compute_spin_orbit_operator
  !
  ! ====================================================================================
  subroutine init_frpp_parameters ( )
    ! ----------------------------------------------------------------------------------
    
    USE ions_base,             ONLY : nsp, nat, ityp
    USE uspp_param,            ONLY : lmaxkb, nh, nbetam, nhm, lmaxq
    USE uspp,                  ONLY : nkb, nkbus
    USE io_global,             ONLY : stdout
    USE us,                    ONLY : qrad, tab, tab_d2y, dq, nqx, nqxq, spline_ps
    USE cellmd,                ONLY : cell_factor
    USE gvect,                 ONLY : gcutm
    USE gvecw,                 ONLY : ecutwfc
    USE klist,                 ONLY : qnorm
    
    implicit none
    
    !
    !   internal variables
    
    INTEGER                         :: na, nb, nt
    
    !
    lmaxkb = -1
    do nt= 1, nsp
       !
       nh (nt) = 0
       !
       ! do not add any beta projector if pseudo in 1/r fmt (AF)
       IF ( frpp(nt)%tcoulombp ) CYCLE
       !
       do nb= 1, frpp(nt)%nbeta
          nh (nt) = nh (nt) + 2 * frpp(nt)%lll(nb) + 1
          lmaxkb = MAX ( lmaxkb, frpp(nt)%lll(nb) )
       end do
       !
    end do
    !
    !   compute max number beta functions
    !
    nhm = MAXVAL ( nh (1:nsp) )
    nbetam = MAXVAL (frpp(:)%nbeta)
    
    !
    ! calculate the number of beta functions of the solid
    !
    nkb = 0
    nkbus = 0
    do na= 1, nat
       nt = ityp(na)
       nkb = nkb + nh (nt)
       if (frpp(nt)%tvanp) nkbus = nkbus + nh (nt)
    end do
    !
    IF ( spline_ps .AND. cell_factor <= 1.1d0 ) cell_factor = 1.1d0
    nqxq = INT ( ( (SQRT(gcutm) + qnorm) / dq + 4) * cell_factor )
    lmaxq = 2*lmaxkb+1
    !
    IF ( ALLOCATED (qrad) ) DEALLOCATE (qrad)
    IF (lmaxq > 0) ALLOCATE ( qrad (nqxq, nbetam*(nbetam+1)/2, lmaxq, nsp) )
    !
    nqx = INT ( (SQRT(ecutwfc) / dq + 4) * cell_factor )
    !
    IF ( ALLOCATED (tab) ) DEALLOCATE (tab)
    ALLOCATE ( tab (nqx,nbetam,nsp) )
    !
    !  d2y
    IF ( ALLOCATED (tab_d2y) ) DEALLOCATE (tab_d2y)
    IF (spline_ps) ALLOCATE ( tab_d2y (nqx,nbetam,nsp) )
    !
    
    call allocate_FR_pp_variables ()
    
    RETURN
    !
  end subroutine init_frpp_parameters
  !
  ! ====================================================================================
  subroutine allocate_FR_pp_variables ( )
    ! ----------------------------------------------------------------------------------
    
    USE io_global,             ONLY : stdout
    USE uspp_param,            ONLY : nhm, nh
    USE ions_base,             ONLY : nsp, nat
    USE spin_orb,              ONLY : fcoef
    USE uspp,                  ONLY : dvan_so, qq_so, qq_at, qq_nt, nhtol, nhtolm, indv, nhtoj, ijtoh, indv_ijkb0
    
    !
    implicit none
    
    !        internal variables
    !
    
    !        the SOC arrays should not be allocated here
    IF ( ALLOCATED (fcoef) ) DEALLOCATE (fcoef)
    ALLOCATE ( fcoef (nhm,nhm,2,2,nsp) )
    !
    IF ( ALLOCATED (dvan_so) ) DEALLOCATE (dvan_so)
    ALLOCATE ( dvan_so (nhm,nhm,4,nsp) )
    !
    IF ( ALLOCATED (qq_so) ) DEALLOCATE (qq_so)
    ALLOCATE ( qq_so (nhm,nhm,4,nsp) )
    !
    IF ( ALLOCATED ( qq_at ) ) DEALLOCATE ( qq_at )
    ALLOCATE ( qq_at (nhm,nhm,nat) )
    !
    IF ( ALLOCATED ( qq_nt ) ) DEALLOCATE ( qq_nt )
    ALLOCATE ( qq_nt (nhm,nhm,nsp) )
    !
    IF ( ALLOCATED ( nhtol ) ) DEALLOCATE ( nhtol )
    ALLOCATE ( nhtol (nhm,nsp) )
    !
    IF ( ALLOCATED ( nhtolm ) ) DEALLOCATE ( nhtolm )
    ALLOCATE ( nhtolm (nhm,nsp) )
    !
    IF ( ALLOCATED ( indv ) ) DEALLOCATE ( indv )
    ALLOCATE ( indv (nhm,nsp) )
    !
    IF ( ALLOCATED ( nhtoj ) ) DEALLOCATE ( nhtoj )
    ALLOCATE ( nhtoj (nhm,nsp) )
    !
    IF ( ALLOCATED ( ijtoh ) ) DEALLOCATE ( ijtoh )
    ALLOCATE ( ijtoh (nhm,nhm,nsp) )
    !
    IF ( ALLOCATED ( indv_ijkb0 ) ) DEALLOCATE ( indv_ijkb0 )
    ALLOCATE ( indv_ijkb0 (nat) )
    
    !
    RETURN
    !
  end subroutine allocate_FR_pp_variables
  !
  ! ====================================================================================
  SUBROUTINE read_FR_pseudo_from_file ( unit )
    ! ----------------------------------------------------------------------------------
    
    USE io_global,              ONLY : ionode, stdout
    USE parser,                 ONLY : parse_unit, read_line
    
    !
    IMPLICIT NONE
    
    !    internal variables
    !
    INTEGER, INTENT(IN), OPTIONAL   :: unit
    !
    CHARACTER(LEN=256)              :: input_line
    CHARACTER(LEN=80)               :: card
    CHARACTER(LEN=1), EXTERNAL      :: capital
    !
    INTEGER                         :: i
    !
    LOGICAL                         :: tend
    
    !
    ! read_line reads from unit parse_unit
    !
    IF (present(unit)) THEN
       parse_unit =  unit
    ELSE
       parse_unit =  5
    END IF
    !
100 CALL read_line( input_line, end_of_file=tend )
    !
    IF( tend ) GOTO 120
    IF( input_line == ' ' .OR. input_line(1:1) == '#' .OR. &
         input_line(1:1) == '!' ) GOTO 100
    !
    READ (input_line, *) card
    !
    DO i = 1, len_trim( input_line )
       input_line( i : i ) = capital( input_line( i : i ) )
    ENDDO
    !
    IF ( trim(card) == 'ATOMIC_PSEUDO' ) THEN
       !
       CALL card_atomic_pseudo( input_line )
       !
    ELSE
       !
       IF ( ionode ) &
            WRITE( stdout,'(A)') 'Warning: card '//trim(input_line)//' ignored'
       !
    ENDIF
    !
    ! ... END OF LOOP ... !
    !
    GOTO 100
    !
120 CONTINUE
    !
    RETURN
    !
  END SUBROUTINE read_FR_pseudo_from_file
  !
  ! ===================================================================================
  SUBROUTINE card_atomic_pseudo( input_line )
    !
    USE ions_base,       ONLY : ntyp => nsp
    USE parser,          ONLY : read_line
    USE io_global,       ONLY : stdout
    
    !
    IMPLICIT NONE
    !
    CHARACTER(len=256)        :: input_line
    INTEGER                   :: is, ip, ierr
    CHARACTER(len=4)          :: lb_pos
    CHARACTER(len=256)        :: psfile
    CHARACTER(len=3)          :: atom_label(ntypx) = 'XX'
    !
    !
    IF ( taspc ) THEN
       CALL errore( ' card_atomic_pseudo  ', ' two occurrences', 2 )
    ENDIF
    IF ( ntyp > ntypx ) THEN
       CALL errore( ' card_atomic_pseudo ', ' nsp out of range ', ntyp )
    ENDIF
    !
    
    DO is = 1, ntyp
       !
       CALL read_line( input_line )
       READ( input_line, *, iostat=ierr ) lb_pos, psfile
       CALL errore( ' card_atomic_pseudo ', &
            'cannot read atomic specie from: '//trim(input_line), abs(ierr))
       frpsfile(is)   = trim( psfile )
       lb_pos         = adjustl( lb_pos )
       atom_label(is) = trim( lb_pos )
       !
       DO ip = 1, is - 1
          IF ( atom_label(ip) == atom_label(is) ) THEN
             CALL errore( ' card_atomic_pseudo ', &
                  & ' two occurrences of the same atomic label ', is )
          ENDIF
       ENDDO
       !
    ENDDO
    taspc = .true.
    !
    RETURN
    !
  END SUBROUTINE card_atomic_pseudo
  !
  ! ====================================================================================
  SUBROUTINE read_FR_pseudo (input_dft, printout, ecutwfc_pp, ecutrho_pp)
    ! ----------------------------------------------------------------------------------
    !
    USE radial_grids,                        ONLY : nullify_radial_grid
    USE pseudo_types,                        ONLY : nullify_pseudo_upf
    USE ions_base,                           ONLY : ntyp => nsp, zv
    USE io_global,                           ONLY : ionode, stdout, ionode_id
    USE io_files,                            ONLY : pseudo_dir, pseudo_dir_cur, tmp_dir
    USE mp_images,                           ONLY : intra_image_comm
    USE funct,                               ONLY : get_inlc, get_igcc, get_igcx, get_icorr, get_iexch, set_dft_from_name,  &
                                                    enforce_input_dft
    USE uspp,                                ONLY : nlcc_any, okvan
    USE uspp_param,                          ONLY : nvb
    USE wrappers,                            ONLY : f_remove, md5_from_file
    USE upf_module,                          ONLY : read_upf
    USE upf_to_internal,                     ONLY : set_upf_q, add_upf_grid
    USE emend_upf_module,                    ONLY : make_emended_upf_copy
    USE mp,                                  ONLY : mp_sum, mp_bcast
    USE read_uspp_module,                    ONLY : readvan, readrrkj
    USE m_gth,                               ONLY : readgth
    
    !
    implicit none
    !
    CHARACTER(len=*), INTENT(INOUT)                 :: input_dft
    LOGICAL, OPTIONAL, INTENT(IN)                   :: printout
    real(DP), OPTIONAL, INTENT(OUT)                 :: ecutwfc_pp, ecutrho_pp
    !  information on atomic radial grid
    !
    REAL(DP), parameter :: rcut = 10.d0 
    ! 2D Coulomb cutoff: modify this (at your own risks) if problems with cutoff 
    ! being smaller than pseudo rcut. original value=10.0
    character(len=256)                              :: file_pseudo
    character(len=256)                              :: file_fixed, msg
    integer                                         :: iunps, ios, isupf
    integer                                         :: nt, ir
    INTEGER                                         :: icorr_, iexch_, igcc_, igcx_, inlc_
    !
    LOGICAL                                         :: printout_ = .false.
    logical                                         :: exst, is_xml
    
    !
    !  ... allocate local radial grid
    !
    iunps = 4
    !
    ALLOCATE ( rgrid (ntyp), msh (ntyp) )

    !
    DO nt= 1, ntyp
       call nullify_radial_grid ( rgrid (nt) )
    END DO

    !
    ALLOCATE ( frpp (ntyp) )
    !
    DO nt= 1, ntyp
       call nullify_pseudo_upf ( frpp (nt) )
    END DO

    !
    IF ( PRESENT (printout) ) THEN
       printout_ = printout .AND. ionode
    END IF
    IF ( printout_ ) THEN
       WRITE( stdout,"(//,3X,'Atomic FR Pseudopotentials Parameters',/, &
            &    3X,'----------------------------------' )" )
    END IF
    !
    DO nt= 1, ntyp
       !
       !   variables not necessary for USPP, but necessary for PAW
       !
       rgrid(nt)%xmin = 0.d0
       rgrid(nt)%dx = 0.d0
       !
       !   try first pseudo_dir_cur if set: in case of restart from file,
       !   this is where PP files should be located
       !
       ios = 1
       IF ( pseudo_dir_cur /= ' ' ) THEN
          file_pseudo  = TRIM (pseudo_dir_cur) // TRIM (frpsfile(nt))
          INQUIRE(file = file_pseudo, EXIST = exst) 
          IF (exst) ios = 0
          CALL mp_sum (ios,intra_image_comm)
          IF ( ios /= 0 ) CALL infomsg &
               ('read_FR_pseudo', 'file '//TRIM(file_pseudo)//' not found')
          !
          ! file not found? no panic (yet): if the restart file is not visible
          ! to all processors, this may happen. Try the original location
       END IF
       !
       ! try the original location pseudo_dir, as set in input
       ! (it should already contain a slash at the end)
       !
       IF ( ios /= 0 ) THEN
          file_pseudo = TRIM (pseudo_dir) // TRIM (frpsfile(nt))
          INQUIRE ( file = file_pseudo, EXIST = exst) 
          IF (exst) ios = 0
          CALL mp_sum (ios,intra_image_comm)
          CALL errore('read_FR_pseudo', 'file '//TRIM(file_pseudo)//' not found',ABS(ios))
       END IF
       !
       frpp(nt)%grid => rgrid(nt)
       !
       IF (printout_) THEN
          WRITE( stdout, "(/,3X,'Reading pseudopotential for specie # ',I2, &
               & ' from file :',/,3X,A)") nt, TRIM(file_pseudo)
       END IF
       !
       isupf = 0
       CALL read_upf(frpp(nt), rgrid(nt), isupf, filename = file_pseudo )
       !!
       !! start reading - check  first if files are readable as xml files,
       !! then as UPF v.2, then as UPF v.1
       !
       IF (isupf ==-81 ) THEN
          !! error -81 may mean that file contains offending characters
          !! fix and write file to tmp_dir (done by a single processor)
          file_fixed = TRIM(tmp_dir)//TRIM(frpsfile(nt))//'_'
          !! the underscore is added to distinguish this "fixed" file 
          !! from the original one, in case the latter is in tmp_dir
          !
          IF ( ionode ) is_xml = make_emended_upf_copy( file_pseudo, file_fixed ) 
          CALL mp_bcast (is_xml,ionode_id,intra_image_comm)
          !
          IF (is_xml) THEN
             !
             CALL  read_upf(frpp(nt), rgrid(nt), isupf, filename = TRIM(file_fixed) )
             !! try again to read from the corrected file 
             WRITE ( msg, '(A)') 'Pseudo file '// trim(frpsfile(nt)) // ' has been fixed on the fly.' &
                  // new_line('a') // 'To avoid this message in the future, permanently fix ' &
                  // new_line('a') // ' your pseudo files following these instructions: ' &
                  // new_line('a') // 'https://gitlab.com/QEF/q-e/blob/master/upftools/how_to_fix_upf.md'
             CALL infomsg('read_upf:', trim(msg) )    
          ELSE
             !
             OPEN ( UNIT = iunps, FILE = file_pseudo, STATUS = 'old', FORM = 'formatted' ) 
             CALL  read_upf(frpp(nt), rgrid(nt), isupf, UNIT = iunps )
             !! try to read UPF v.1 file
             CLOSE (iunps)
             !
          END IF
          !
          IF (ionode) ios = f_remove( file_fixed )
          !
       END IF
       !
       IF (isupf == -2 .OR. isupf == -1 .OR. isupf == 0) THEN
          !
          IF( printout_) THEN
             IF ( isupf == 0 ) THEN
                WRITE( stdout, "(3X,'file type is xml')") 
             ELSE
                WRITE( stdout, "(3X,'file type is UPF v.',I1)") ABS(isupf) 
             END IF
          END IF
          !
          !  reconstruct Q(r) if required
          !
          call set_upf_q (frpp (nt))
          !
       ELSE
          !
          OPEN (UNIT=iunps, FILE=TRIM(file_pseudo), STATUS='old', FORM='formatted')
          !
          !     The type of the pseudopotential is determined by the file name:
          !    *.xml or *.XML  UPF format with schema              pp_format=0
          !    *.upf or *.UPF  UPF format                          pp_format=1
          !    *.vdb or *.van  Vanderbilt US pseudopotential code  pp_format=2
          !    *.gth           Goedecker-Teter-Hutter NC pseudo    pp_format=3
          !    *.RRKJ3         Andrea's   US new code              pp_format=4
          !    none of the above: PWSCF norm-conserving format     pp_format=5
          !
          IF ( pp_format (frpsfile (nt) ) == 2  ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is Vanderbilt US PP')")
             CALL readvan (iunps, nt, frpp(nt))
             !
          ELSE IF ( pp_format (frpsfile (nt) ) == 3 ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is GTH (analytical)')")
             CALL readgth (iunps, nt, frpp(nt))
             !
          ELSE IF ( pp_format (frpsfile (nt) ) == 4 ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is RRKJ3')")
             CALL readrrkj (iunps, nt, frpp(nt))
             !
          ELSE IF ( pp_format (frpsfile (nt) ) == 5 ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is old PWscf NC format')")
             CALL read_ncpp (iunps, nt, frpp(nt))
             !
          ELSE
             !
             CALL errore('read_FR_pseudo', 'file '//TRIM(file_pseudo)//' not readable',1)
             !
          ENDIF
          !
          ! add grid information, reconstruct Q(r) if needed
          !
          CALL add_upf_grid (frpp(nt), rgrid(nt))
          !
          ! end of reading
          !
          CLOSE (iunps)
          !
       END IF
       !
       ! calculate MD5 checksum for this pseudopotential
       !
       call md5_from_file(file_pseudo, frpp(nt)%md5_cksum)
       !
    END DO
    !
    !  end PP reading -> set some variables
    !
    IF (input_dft /= 'none') call enforce_input_dft (input_dft)
    !
    nvb = 0
    DO nt= 1, ntyp
       !
       ! ... Zv = valence charge of the (pseudo-)atom, read from PP files,
       ! ... is set equal to Zp = pseudo-charge of the pseudopotential
       !
       zv (nt) = frpp (nt)%zp
       ! count US species
       IF ( frpp(nt)%tvanp ) nvb = nvb + 1
       !
       ! ... set DFT value
       !
       call set_dft_from_name ( frpp(nt)%dft )
       !
       ! ... Check for DFT consistency - ignored if dft enforced from input
       !
       IF (nt == 1) THEN
          iexch_ = get_iexch()
          icorr_ = get_icorr()
          igcx_  = get_igcx()
          igcc_  = get_igcc()
          inlc_  = get_inlc()
       ELSE
          IF ( iexch_ /= get_iexch() .OR. icorr_ /= get_icorr() .OR. &
               igcx_  /= get_igcx()  .OR. igcc_  /= get_igcc()  .OR.  &
               inlc_  /= get_inlc() ) THEN
             CALL errore( 'read_FR_pseudo','inconsistent DFT read from PP files', nt)
          END IF
       END IF
       !
       ! the radial grid is defined up to r(mesh) but we introduce 
       ! an auxiliary variable msh to limit the grid up to rcut=10 a.u. 
       ! This is used to cut off the numerical noise arising from the
       ! large-r tail in cases like the integration of V_loc-Z/r
       !
       DO ir = 1, rgrid(nt)%mesh
          IF (rgrid(nt)%r(ir) > rcut) THEN
             msh (nt) = ir
             GOTO 5
          END IF
       END DO
       msh (nt) = rgrid(nt)%mesh 
5      msh (nt) = 2 * ( (msh (nt) + 1) / 2) - 1
       !
       ! msh is forced to be odd for simpson integration (maybe obsolete?)
       !
       ! check for zero atomic wfc, 
       ! check that (occupied) atomic wfc are properly normalized
       !
       CALL check_atwfc_norm(nt)
       !
    END DO
    !
    ! more initializations
    !
    okvan = ( nvb > 0 )
    nlcc_any = ANY ( frpp(1:ntyp)%nlcc )
    !
    ! return cutoff read from PP file, if required
    !
    IF ( PRESENT(ecutwfc_pp) ) THEN
       ecutwfc_pp = MAXVAL ( frpp(1:ntyp)%ecutwfc )
    END IF
    IF ( PRESENT(ecutrho_pp) ) THEN
       ecutrho_pp = MAXVAL ( frpp(1:ntyp)%ecutrho )
    END IF
    !
    WRITE(stdout,*) 'ALL SET'
    !
  END SUBROUTINE read_FR_pseudo
  !
  !---------------------------------------------------------------
  SUBROUTINE check_atwfc_norm(nt)
    !---------------------------------------------------------------
    !  check for the presence of zero wavefunctions first
    !  check the normalization of the atomic wfc (only those with non-negative
    !  occupations) and renormalize them if the calculated norm is incorrect 
    !  by more than eps6 (10^{-6})
    !
    USE constants,    ONLY : eps6, eps8
    USE io_global,    ONLY : stdout
    
    implicit none
    
    integer,intent(in) :: nt ! index of the pseudopotential to be checked
    !
    integer ::             &
         mesh, kkbeta,       & ! auxiliary indices of integration limits
         l,                  & ! orbital angular momentum 
         iwfc, ir,           & ! counter on atomic wfcs and on radial mesh
         ibeta, ibeta1, ibeta2 ! counters on betas
    logical :: &
         match                 ! a logical variable 
    real(DP) :: &
         norm,               & ! the norm
         j                     ! total (spin+orbital) angular momentum
    real(DP), allocatable :: &
         work(:), gi(:)        ! auxiliary variable for becp
    character (len=80) :: renorm
    !
    allocate (work(frpp(nt)%nbeta), gi(frpp(nt)%grid%mesh) )
    
    ! define indices for integration limits
    mesh = frpp(nt)%grid%mesh
    kkbeta = frpp(nt)%kkbeta
    !
    renorm = ' '
    DO iwfc = 1, frpp(nt)%nwfc
       l = frpp(nt)%lchi(iwfc)
       if ( frpp(nt)%has_so ) j = frpp(nt)%jchi(iwfc)
       !
       ! the smooth part first ..
       gi(1:mesh) = frpp(nt)%chi(1:mesh,iwfc) * frpp(nt)%chi(1:mesh,iwfc)
       call simpson (mesh, gi, frpp(nt)%grid%rab, norm)
       !
       IF ( norm < eps8 ) then
          WRITE( stdout,'(5X,"WARNING: atomic wfc # ",i2, &
               & " for atom type",i2," has zero norm")') iwfc, nt
          !
          ! set occupancy to a small negative number so that this wfc
          ! is not going to be used for starting wavefunctions
          !
          frpp(nt)%oc (iwfc) = -eps8
       END IF
       !
       IF ( frpp(nt)%oc(iwfc) < 0.d0) CYCLE ! only occupied states are normalized
       !
       if (  frpp(nt)%tvanp ) then
          !
          ! the US part if needed
          do ibeta = 1, frpp(nt)%nbeta
             match = l.eq.frpp(nt)%lll(ibeta)
             if (frpp(nt)%has_so) match=match.and.abs(j-frpp(nt)%jjj(ibeta)) < eps6
             if (match) then
                gi(1:kkbeta)= frpp(nt)%beta(1:kkbeta,ibeta) * &
                     frpp(nt)%chi (1:kkbeta,iwfc) 
                call simpson (kkbeta, gi, frpp(nt)%grid%rab, work(ibeta))
             else
                work(ibeta)=0.0_dp
             endif
          enddo
          do ibeta1=1,frpp(nt)%nbeta
             do ibeta2=1,frpp(nt)%nbeta
                norm=norm+frpp(nt)%qqq(ibeta1,ibeta2)*work(ibeta1)*work(ibeta2)  
             enddo
          enddo
       end if
       norm=sqrt(norm)
       if (abs(norm-1.0_dp) > eps6 ) then
          renorm = TRIM(renorm) // ' ' // frpp(nt)%els(iwfc)
          frpp(nt)%chi(1:mesh,iwfc)=frpp(nt)%chi(1:mesh,iwfc)/norm
       end if
    end do
    deallocate (work, gi )
    IF ( LEN_TRIM(renorm) > 0 ) WRITE( stdout, &
         '(15x,"file ",a,": wavefunction(s) ",a," renormalized")') &
         TRIM(frpsfile(nt)),TRIM(renorm)
    RETURN
    !
  END SUBROUTINE check_atwfc_norm
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION pp_format (psfile)
    !-----------------------------------------------------------------------
    IMPLICIT NONE
    CHARACTER (LEN=*) :: psfile
    INTEGER :: l
    !
    l = LEN_TRIM (psfile)
    pp_format = 5
    IF (l > 3) THEN
       IF (psfile (l-3:l) =='.xml' .OR. psfile (l-3:l) =='.XML') THEN
          pp_format = 0
       ELSE IF (psfile (l-3:l) =='.upf' .OR. psfile (l-3:l) =='.UPF') THEN
          pp_format = 1
       ELSE IF (psfile (l-3:l) =='.vdb' .OR. psfile (l-3:l) =='.van') THEN
          pp_format = 2
       ELSE IF (psfile (l-3:l) =='.gth') THEN
          pp_format = 3
       ELSE IF (l > 5) THEN
          If (psfile (l-5:l) =='.RRKJ3') pp_format = 4
       END IF
    END IF
    !
  END FUNCTION pp_format
  !
  ! =======================================================
  SUBROUTINE set_spin_orbit_operator ()
    ! -----------------------------------------------------
    
    USE constants,        ONLY : sqrt2, fpi
    USE uspp_param,       ONLY : nh, lmaxkb, lmaxq, nbetam, nhm
    USE parameters,       ONLY : lmaxx
    USE ions_base,        ONLY : ntyp => nsp, nat, ityp
    USE io_global,        ONLY : stdout
    USE cell_base,        ONLY : omega, tpiba
    USE gvect,            ONLY : g, gg
    USE us,               ONLY : qrad, tab, tab_d2y, dq, nqx, nqxq, spline_ps
    USE uspp,             ONLY : indv, qq_at, qq_nt, qq_so, ap, ijtoh, dvan_so, okvan,   &
                                 nhtol, nhtolm, nhtoj, indv_ijkb0, aainit
    USE spin_orb,         ONLY : fcoef, rot_ylm
    USE mp_bands,         ONLY : intra_bgrp_comm
    USE mp,               ONLY : mp_sum
    USE paw_variables,    ONLY : okpaw
    USE splinelib,        ONLY : spline
    
    !
    implicit none
    
    !    internal variables
    
    integer                      :: nt, ih, jh, ia, ijv, iq, ir, is, startq, lastq, ndm, nb, mb
    integer                      :: l, m
    integer                      :: vi, vj, vk
    integer                      :: n, n1, ijkb0, ijs, is1, is2, kh, lh, li, lk, m0, m1, na, mi, mk
    integer, external            :: sph_ind
    !
    real(DP), allocatable        :: aux (:), aux1 (:), qtot (:,:), besr (:)
    !  various work space
    real(DP), allocatable        :: xdata (:)
    real(DP), allocatable        :: ylmk0 (:)
    real(DP)                     :: ji, jk, d1
    real(DP)                     :: pref, prefr, q, qi
    !  sph. harmonics
    complex(DP)                  :: coeff, qgm(1)
    real(DP)                     :: j, vqint
    real(DP), EXTERNAL           :: spinor
    
    call start_clock ('set_spin_orbit_operator')
    !
    !    variables initialization
    !
    ndm = MAXVAL ( frpp(:)%kkbeta )
    allocate ( aux (ndm) )
    allocate ( aux1(ndm) )
    allocate ( ylmk0(lmaxq*lmaxq) )
    allocate ( qtot(ndm, nbetam*(nbetam+1)/2) )
    ap (:,:,:) = 0.d0
    if (lmaxq > 0) qrad (:,:,:,:) = 0.d0
    !
    ! the following prevents an out-of-bound error: frpp(nt)%nqlc=2*lmax+1
    ! but in some versions of the PP files lmax is not set to the maximum
    ! l of the beta functions but includes the l of the local potential
    !
    do nt= 1, ntyp
       frpp(nt)%nqlc = MIN ( frpp(nt)%nqlc, lmaxq )
       IF ( frpp(nt)%nqlc < 0 ) frpp(nt)%nqlc = 0
    end do
    !
    prefr = fpi / omega
    
    !
    !  In the spin orbit case we need the unitary matrix u which rotates the
    !  real spherical harmonics and yields the complex ones
    !
    rot_ylm = (0.d0, 0.d0)
    l = lmaxx
    rot_ylm(l+1,1) = (1.d0,0.d0)
    do n1=2,2*l+1,2
       m=n1/2
       n=l+1-m
       rot_ylm(n,n1) = cmplx ((-1.d0)**m/sqrt2,0.d0,kind=dp)
       rot_ylm(n,n1+1) = cmplx (0.d0,-(-1.d0)**m/sqrt2,kind=dp)
       n=l+1+m
       rot_ylm(n,n1) = cmplx (1.d0/sqrt2,0.d0,kind=dp)
       rot_ylm(n,n1+1) = cmplx (0.d0,1.d0/sqrt2,kind=dp)
    end do
    fcoef = (0.d0,0.d0)
    dvan_so = (0.d0,0.d0)
    qq_so = (0.d0,0.d0)
    qq_at = 0.d0
    qq_nt = 0.d0
    !
    !   For each pseudopotential we initialize the indices nhtol, nhtolm,
    !   nhtoj, indv, and if the pseudopotential is of KB type we initialize the
    !   atomic D terms
    !
    ijkb0 = 0
    do nt= 1, ntyp
       ih = 1
       do nb= 1, frpp(nt)%nbeta
          l = frpp(nt)%lll (nb)
          do m= 1, 2*l+1
             nhtol (ih,nt) = l
             nhtolm(ih,nt) = l*l+m
             indv (ih,nt)  = nb
             ih = ih+1
          end do
       end do
       if ( frpp(nt)%has_so ) then
          ih = 1
          do nb= 1, frpp(nt)%nbeta
             l = frpp(nt)%lll (nb)
             j = frpp(nt)%jjj (nb)
             do m= 1, 2*l+1
                nhtoj (ih,nt) = j
                ih = ih+1
             end do
          end do
       end if
       !
       ! ijtoh map augmentation channel indexes ih and jh to composite
       ! "triangular" index ijh
       ijtoh (:,:,nt) = -1
       ijv = 0
       do ih= 1, nh(nt)
          do jh= ih, nh(nt)
             ijv = ijv + 1
             ijtoh (ih,jh,nt) = ijv
             ijtoh (jh,ih,nt) = ijv
          end do
       end do
       !
       ! ijkb0 is just before the first beta "in the solid" for atom ia
       ! i.e. ijkb0+1,.. ijkb0+nh(ityp(ia)) are the nh beta functions of
       !      atom ia in the global list of beta functions
       do ia= 1, nat
          IF ( ityp(ia) == nt ) THEN
             indv_ijkb0 (ia) = ijkb0
             ijkb0 = ijkb0 + nh (nt)
          END IF
       end do
       !
       !    From now on the only difference between KB and US pseudopotentials
       !    is in the presence of the q and Q functions.
       !
       !    Here we initialize the D of the solid
       !
       IF ( frpp(nt)%has_so ) THEN
          !
          !  compute fcoef
          !
          do ih= 1, nh(nt)
             li = nhtol (ih,nt)
             ji = nhtoj (ih,nt)
             mi = nhtolm (ih,nt) - li*li
             vi = indv (ih,nt)
             do kh= 1, nh(nt)
                lk = nhtol (kh,nt)
                jk = nhtoj (kh,nt)
                mk = nhtolm(kh,nt) - lk*lk
                vk = indv (kh,nt)
                if (li == lk .and. abs(ji-jk) < 1.d-7) then
                   do is1= 1, 2
                      do is2= 1, 2
                         coeff = (0.d0,0.d0)
                         do m=-li-1, li
                            m0 = sph_ind(li,ji,m,is1) + lmaxx + 1
                            m1 = sph_ind(lk,jk,m,is2) + lmaxx + 1
                            coeff = coeff + rot_ylm(m0,mi) * spinor(li,ji,m,is1) *   &
                                 CONJG(rot_ylm(m1,mk)) * spinor(lk,jk,m,is2)
                         end do
                         fcoef (ih,kh,is1,is2,nt) = coeff
                      end do
                   end do
                end if
             end do
          end do
          !
          !   compute the bare coefficients
          !
          do ih= 1, nh(nt)
             vi = indv (ih,nt)
             do jh= 1, nh(nt)
                vj = indv (jh,nt)
                ijs = 0
                do is1=1, 2
                   do is2=1, 2
                      ijs = ijs + 1
                      dvan_so (ih,jh,ijs,nt) = frpp(nt)%dion(vi,vj) *      &
                           fcoef (ih,jh,is1,is2,nt)
                      if (vi .ne. vj) fcoef(ih,jh,is1,is2,nt) = (0.d0,0.d0)
                   end do
                end do
             end do
          end do
       ELSE
          do ih= 1, nh(nt)
             do jh= 1, nh(nt)
                if (nhtol (ih,nt) == nhtol (jh,nt) .and.     &
                     nhtolm(ih,nt) == nhtolm(jh,nt) ) then
                   ir = indv (ih,nt)
                   is = indv (jh,nt)
                   dvan_so (ih,jh,1,nt) = frpp(nt)%dion (ir,is)
                   dvan_so (ih,jh,4,nt) = frpp(nt)%dion (ir,is)
                end if
             end do
          end do
       END IF
    end do
    !
    !    compute Clebsch-Gordan coefficients
    !
    if (okvan .or. okpaw) call aainit (lmaxkb+1)
    !
    !   here for the US types we compute the Fourier transform of the
    !   Q functions.
    !
    call divide (intra_bgrp_comm, nqxq, startq, lastq)
    !
    do nt= 1, ntyp
       if ( frpp(nt)%tvanp ) then
          do l= 0, frpp(nt)%nqlc - 1
             !
             !     first we build for each nb,mb,l the total Q(|r|) function
             !     note that l is the true (combined) angular momentum
             !     and that the arrays have dimensions 0..l (no more 1..l+1)
             !
             do nb= 1, frpp(nt)%nbeta
                do mb= nb, frpp(nt)%nbeta
                   respect_sum_rule : if ( ( l >= abs(frpp(nt)%lll(nb) - frpp(nt)%lll(mb)) ) .and.  &
                        ( l <= frpp(nt)%lll(nb) + frpp(nt)%lll(mb) ) .and. &
                        (mod (l+frpp(nt)%lll(nb)+frpp(nt)%lll(mb), 2) == 0) ) then
                      ijv = mb * (mb-1) / 2 + nb
                      ! in PAW and now in US as well q(r) is stored in an l-dependent array
                      qtot(1:frpp(nt)%kkbeta,ijv) = frpp(nt)%qfuncl(1:frpp(nt)%kkbeta,ijv,l)
                   endif respect_sum_rule
                end do
             end do
             !
             !     here we compute the spherical bessel function for each |g|
             !
             do iq= startq, lastq
                q = (iq-1) * dq * tpiba
                call sph_bes ( frpp(nt)%kkbeta, rgrid(nt)%r, q, l, aux)
                !
                !   and then we integrate with all the Q functions
                !
                do nb= 1, frpp(nt)%nbeta
                   !
                   !    the Q are symmetric with respect to indices
                   !
                   do mb = nb, frpp(nt)%nbeta
                      ijv = mb * (mb - 1) / 2 + nb
                      if ( ( l >= abs(frpp(nt)%lll(nb) - frpp(nt)%lll(mb)) ) .and. &
                           ( l <=     frpp(nt)%lll(nb) + frpp(nt)%lll(mb)  ) .and. &
                           (mod(l+frpp(nt)%lll(nb)+frpp(nt)%lll(mb),2)==0) ) then
                         do ir = 1, frpp(nt)%kkbeta
                            aux1 (ir) = aux (ir) * qtot (ir, ijv)
                         enddo
                         call simpson ( frpp(nt)%kkbeta, aux1, rgrid(nt)%rab, &
                              qrad(iq,ijv,l + 1, nt) )
                      endif
                   enddo
                end do
                ! igl
             end do
             ! l
          end do
          qrad (:,:,:,nt) = qrad (:,:,:,nt) * prefr
          call mp_sum ( qrad (:,:,:,nt), intra_bgrp_comm )
       end if
       ! ntyp
    end do
    deallocate (aux1)
    deallocate (qtot)
    !
    !   and finally we compute the qq coefficients by integrating the Q.
    !   q are the g=0 components of Q.
    !
#if defined(__MPI)
    if (gg (1) > 1.0d-8) goto 100
#endif
    call ylmr2 (lmaxq*lmaxq,1,g,gg,ylmk0)
    do nt= 1, ntyp
       if (frpp(nt)%tvanp) then
          if (frpp(nt)%has_so) then
             do ih= 1, nh(nt)
                do jh= 1, nh(nt)
                   call qvan2 (1, ih, jh, nt, gg, qgm, ylmk0)
                   qq_nt (ih,jh,nt) = omega * DBLE( qgm(1) )
                   do kh= 1, nh(nt)
                      do lh= 1, nh(nt)
                         ijs=0
                         do is1= 1, 2
                            do is2= 1, 2
                               ijs=ijs+1
                               do is= 1, 2
                                  qq_so (kh,lh,ijs,nt) = qq_so (kh,lh,ijs,nt)           &
                                       + omega * DBLE(qgm(1)) * fcoef(kh,ih,is1,is,nt)  &
                                       * fcoef(jh,lh,is,is2,nt)
                               end do
                            end do
                         end do
                      end do
                   end do
                end do
             end do
          else
             do ih= 1, nh(nt)
                do jh= ih, nh(nt)
                   call qvan2 (1, ih, jh, nt, gg, qgm, ylmk0)
                   qq_so (ih,jh,1,nt) = omega * DBLE (qgm(1))
                   qq_so (jh,ih,1,nt) = qq_so (ih,jh,1,nt)
                   qq_so (ih,jh,4,nt) = qq_so (ih,jh,1,nt)
                   qq_so (jh,ih,4,nt) = qq_so (ih,jh,4,nt)
                   qq_nt (ih,jh,nt) = omega * DBLE (qgm (1))
                   qq_nt (jh,ih,nt) = omega * DBLE (qgm (1))
                end do
             end do
          end if
       end if
    end do
#if defined(__MPI)
100 continue
    call mp_sum ( qq_so, intra_bgrp_comm )
    call mp_sum ( qq_nt, intra_bgrp_comm )
#endif
    ! finally we set the atomic specific qq_at matrices
    do na= 1, nat
       qq_at (:,:,na) = qq_nt (:,:,ityp(na))
    end do
    deallocate (ylmk0)
    deallocate (aux)
    !
    !    fill the interpolation table tab
    !
    allocate ( aux(ndm) )
    allocate ( besr(ndm) )
    pref = fpi / sqrt (omega)
    call divide (intra_bgrp_comm, nqx, startq, lastq)
    tab (:,:,:) = 0.d0
    do nt= 1, ntyp
       if (frpp(nt)%is_gth) cycle
       do nb= 1, frpp(nt)%nbeta
          l = frpp(nt)%lll (nb)
          do iq= startq, lastq
             qi = (iq-1) * dq
             call sph_bes (frpp(nt)%kkbeta, rgrid(nt)%r, qi, l, besr)
             do ir= 1, frpp(nt)%kkbeta
                aux (ir) = frpp(nt)%beta (ir,nb) * besr (ir) * rgrid(nt)%r(ir)
             end do
             call simpson (frpp(nt)%kkbeta, aux, rgrid(nt)%rab, vqint)
             tab (iq,nb,nt) = vqint * pref
          end do
       end do
    end do
    !
    call mp_sum ( tab, intra_bgrp_comm )
    ! initialize spline interpolation
    if (spline_ps) then
       allocate( xdata(nqx) )
       do iq = 1, nqx
          xdata(iq) = (iq - 1) * dq
       enddo
       do nt = 1, ntyp
          do nb = 1, frpp(nt)%nbeta 
             d1 = (tab(2,nb,nt) - tab(1,nb,nt)) / dq
             call spline(xdata, tab(:,nb,nt), 0.d0, d1, tab_d2y(:,nb,nt))
          enddo
       enddo
       deallocate(xdata)
    endif
    
    deallocate (besr)
    deallocate (aux)
    
    call stop_clock ('set_spin_orbit_operator')
    WRITE(stdout,*) "PSEUDO OK"
    return
    !
  END SUBROUTINE set_spin_orbit_operator
  !
  ! =======================================================================
  SUBROUTINE dvan_so_pauli_basis ()
    ! ---------------------------------------------------------------------
    !      transform dvan_so from |up,dw> basis to
    !      basis of pauli matrices
    !
    
    USE spin_orb,            ONLY : fcoef
    USE uspp_param,          ONLY : nh, nhm
    USE uspp,                ONLY : indv
    USE ions_base,           ONLY : nsp
    
    !
    IMPLICIT NONE
    
    !  internal variables
    
    integer                      :: ih, jh, vi, vj
    integer                      :: nt
    integer                      :: is1, is2
    INTEGER                      :: ierr
    
    
    !
    !  allocate Dso
    
    IF (.not. ALLOCATED (Dso) ) THEN
       ALLOCATE ( Dso (nhm,nhm,4,nsp), stat=ierr )
       if (ierr/=0) call errore ('dvan_so_pauli_basis', 'allocating Dso', abs (ierr))
    END IF
    !
    Dso = cmplx (0._dp, 0._dp)
    
    !
    !  iterate over atom species
    !
    
    do nt= 1, nsp
       !
       IF (frpp(nt)%has_so) THEN
          !
          !   compute the bare coefficients
          !
          do ih= 1, nh(nt)
             vi = indv (ih,nt)
             do jh= 1, nh(nt)
                vj = indv (jh,nt)
                !
                ijs = 0
                do is1= 1, 2
                   do is2= 1, 2
                      ijs = ijs + 1
                      Dso (ih,jh,ijs,nt) = frpp(nt)%dion(vi,vj) *    &
                        fcoef (ih,jh,is1,is2,nt)
                      !
                      if (vi .ne. vj) fcoef(ih,jh,is1,is2,nt) = (0.d0,0.d0)
                   end do
                end do
                !
             end do
          end do
          !
       END IF
       !
    end do
    
    !
    RETURN
    !
  END SUBROUTINE dvan_so_pauli_basis

  !
  ! =============================================================================================
  SUBROUTINE compute_soc_matrix_elements ( transitions_list, ntr )
    ! -------------------------------------------------------------------------------------------
    !
    !    this subroutine computes the SOC matrix
    !    elements to be used into the ZFS
    !    calculation
    !
    !    < Psi_o^s | H_SO^a | Psi_n^s' > =
    !             \sum_uv < Psi_o | beta_u > [chi(s) sigma_a chi(s')] <beta_v|Psi_n> Dso_a(u,v)
    !
    
    USE bec_module,           ONLY : bec_sp
    USE ions_base,            ONLY : ntyp => nsp, nat, ityp
    USE uspp_param,           ONLY : nh
    USE control_flags,        ONLY : gamma_only
    USE io_global,            ONLY : stdout
    
    IMPLICIT NONE
    
    integer, intent(in)            :: ntr
    integer, intent(in)            :: transitions_list (ntr,6)
    
    !    internal variables

    integer                        :: a, ih, jh, ikb, jkb, ijkb0, itr, na, nt
    integer                        :: ki, kpi, ni, oi, si, spi
    complex(DP)                    :: s_xyz (3)
    !    spin vector
    INTEGER                        :: ierr
    
    
    
    !
    !    allocate H_SO^a_on
    
    allocate ( HSO_a (ntr, 4), stat=ierr )
    if (ierr/=0) call errore ('compute_soc_matrix_elements', 'allocating HSO_a', abs (ierr))
    HSO_a = (0.d0, 0.d0)
    
    !
    !    iterate over transitions
    !
    
    do itr= 1, ntr
       !
       ki = transitions_list (itr, 1)
       oi = transitions_list (itr, 2)
       si = transitions_list (itr, 3) 
       kpi= transitions_list (itr, 4)
       ni = transitions_list (itr, 5)
       spi= transitions_list (itr, 6) 
       isp= (si-1)*2 + spi
       
       !
       !
       !  iterate atom species
       ijkb0 = 0
       do nt= 1, ntyp
          do na= 1, nat
             IF ( ityp (na) .eq. nt ) THEN
                !
                do ih= 1, nh(nt)
                   ikb = ijkb0 + ih
                   do jh= 1, nh(nt)
                      jkb = ijkb0 + jh
                      !
                      IF (gamma_only) THEN
                         HSO_a (itr,isp) = HSO_a (itr,isp) +     &
                              bec_sp (ki)%r (ikb,oi) * Dso (ih,jh,isp,nt) * bec_sp (kpi)%r (jkb,ni)
                         !
                         !WRITE(stdout,*) HSO_a (itr,a), bec_sp (ki)%r (ikb,oi), s_xyz (a) 
                      ELSE
                         HSO_a (itr,isp) = HSO_a (itr,isp) +     &
                              conjg (bec_sp (ki)%k (ikb,oi)) * Dso (ih,jh,isp,nt) * bec_sp (kpi)%k (jkb,ni)
                         !
                      END IF
                   end do
                end do
                ijkb0 = ijkb0 + nh (nt)
                !
             END IF
          end do
       end do
       !
    end do
    
    !
    RETURN
    !
  END SUBROUTINE compute_soc_matrix_elements
  
  !
END MODULE spin_orbit_operator
