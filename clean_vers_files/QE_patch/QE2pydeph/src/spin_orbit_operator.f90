!
!   MODULE  :  spin orbit operator
!
MODULE spin_orbit_operator
  
  USE kinds,                        ONLY : DP
  USE pseudo_types,                 ONLY : pseudo_upf
  !
  
  TYPE (pseudo_upf), ALLOCATABLE, TARGET :: frpp (:)
  !
  !  fully relativistic pp MUST include SOC
  
  !
CONTAINS
  !
  ! ====================================================================================
  SUBROUTINE read_FR_pseudo (input_dft, printout, ecutwfc_pp, ecutrho_pp)
    ! ----------------------------------------------------------------------------------
    !
    USE radial_grids,                        ONLY : radial_grid_type, nullify_radial_grid
    USE pseudo_types,                        ONLY : nullify_pseudo_upf
    USE ions_base,                           ONLY : ntyp => nsp
    USE io_global,                           ONLY : ionode, stdout
    
    !
    implicit none
    !
    LOGICAL, OPTIONAL, INTENT(IN)                   :: printout
    !  information on atomic radial grid
    type(radial_grid_type), allocatable, target     :: rgrid (:)
    !
    integer, allocatable                            :: msh (:)
    integer                                         :: iunps
    integer                                         :: nt
    !
    LOGICAL                                         :: printout_ = .false.
    
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
       try first pseudo_dir_cur if set: in case of restart from file,
       ! this is where PP files should be located
       !
       ios = 1
       IF ( pseudo_dir_cur /= ' ' ) THEN
          file_pseudo  = TRIM (pseudo_dir_cur) // TRIM (frpsfile(nt))
          INQUIRE(file = file_pseudo, EXIST = exst) 
          IF (exst) ios = 0
          CALL mp_sum (ios,intra_image_comm)
          IF ( ios /= 0 ) CALL infomsg &
               ('readpp', 'file '//TRIM(file_pseudo)//' not found')
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
          CALL errore('readpp', 'file '//TRIM(file_pseudo)//' not found',ABS(ios))
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
          file_fixed = TRIM(tmp_dir)//TRIM(psfile(nt))//'_'
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
             WRITE ( msg, '(A)') 'Pseudo file '// trim(psfile(nt)) // ' has been fixed on the fly.' &
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
          IF ( pp_format (psfile (nt) ) == 2  ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is Vanderbilt US PP')")
             CALL readvan (iunps, nt, upf(nt))
             !
          ELSE IF ( pp_format (psfile (nt) ) == 3 ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is GTH (analytical)')")
             CALL readgth (iunps, nt, upf(nt))
             !
          ELSE IF ( pp_format (psfile (nt) ) == 4 ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is RRKJ3')")
             CALL readrrkj (iunps, nt, upf(nt))
             !
          ELSE IF ( pp_format (psfile (nt) ) == 5 ) THEN
             !
             IF( printout_ ) &
                  WRITE( stdout, "(3X,'file type is old PWscf NC format')")
             CALL read_ncpp (iunps, nt, upf(nt))
             !
          ELSE
             !
             CALL errore('readpp', 'file '//TRIM(file_pseudo)//' not readable',1)
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
             CALL errore( 'readpp','inconsistent DFT read from PP files', nt)
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
  ! =======================================================
  SUBROUTINE set_spin_orbit_operator ()
    ! -----------------------------------------------------

    USE constants,        ONLY : sqrt2
    USE uspp_param,       ONLY : upf, nh
    USE parameters,       ONLY : lmaxx
    USE ions_base,        ONLY : ntyp => nsp
    USE io_global,        ONLY : stdout
    USE uspp,             ONLY : indv, qq_at, qq_nt, qq_so
    USE spin_orb,         ONLY : fcoef, rot_ylm
    
    !
    implicit none
    
    !    internal variables
    
    integer                    :: nt, ih
    integer                    :: vi
    integer                    :: l, m
    integer                    :: n, n1
    
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
       
    !
    !    compute D_so matrix operator
    !
    
    DO nt= 1, ntyp
       
       !
       !  check if upf has SOC
       !
       
       IF (upf(nt)%has_so) THEN
          !
          WRITE (stdout,*) nt, " has SOC"
          
          !
          !   compute bare coeffs
          do ih= 1, nh (nt)
             vi = indv (ih,nt)
          end do
          WRITE(stdout,*) "nbeta= ", upf(nt)%nbeta
          WRITE(stdout,*) shape (upf(nt)%lll)
          WRITE(stdout,*) shape (upf(nt)%jjj)
       ELSE
          !
          WRITE (stdout,*) nt, " has no SOC"
          !
       END IF
       
       
       !
    END DO
    
    
    
    
    
    
    
    
    
    
    
    
    
    RETURN
    !
  END SUBROUTINE set_spin_orbit_operator
  !
  
END MODULE spin_orbit_operator
