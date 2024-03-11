!
!       read full relativistic pseudo file
!       -> store information in frpp
!
! ====================================================================================
SUBROUTINE read_FR_pseudo (frpp, input_dft, printout, ecutwfc_pp, ecutrho_pp)
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
5     msh (nt) = 2 * ( (msh (nt) + 1) / 2) - 1
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
! ===========================================================================
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
!-------------------------------------------------------------------
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
   !--------------------------------------------------------------------
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