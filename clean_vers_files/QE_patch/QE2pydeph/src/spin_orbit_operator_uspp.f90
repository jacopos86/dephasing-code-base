!
!   MODULE  :  set spin orbit operator -> US pseudo potentials
!
MODULE spin_orbit_operator_uspp
   
   USE kinds,                        ONLY : DP
   USE pseudo_types,                 ONLY : pseudo_upf
   USE radial_grids,                 ONLY : radial_grid_type
   USE parameters,                   ONLY : ntypx
   !
   
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
   complex(DP), allocatable                        :: Vso_us (:,:,:,:)
   !
   !  SOC operator

   !
CONTAINS
   !
   ! ====================================================================================
   subroutine init_US_frpp ( frpp )
      ! =================================================================================
      USE ions_base,                         ONLY : nsp

      implicit none
      !
      TYPE (pseudo_upf), INTENT(IN), TARGET       :: frpp (nsp)

      !
      call init_US_frpp_parameters (frpp)

      !
      call allocate_US_frpp_variables ( ) 
      !
   end subroutine init_US_frpp
   !
   ! ====================================================================================
   subroutine compute_spin_orbit_operator_uspp ( frpp )
      ! ----------------------------------------------------------------------------------
      USE ions_base,                        ONLY : nsp

      implicit none

      !
      TYPE (pseudo_upf), intent(inout), TARGET   :: frpp (nsp)
      
      !
      call set_US_spin_orbit_operator (frpp)
      !
      call dvan_so_pauli_basis (frpp)

      !
      RETURN
      !
   end subroutine compute_spin_orbit_operator_uspp
   !
   ! ====================================================================================
   subroutine init_US_frpp_parameters ( frpp )
      ! ---------------------------------------------------------------------------------

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
      TYPE (pseudo_upf), intent(in), TARGET   :: frpp (nsp)

      !
      !   internal variables

      INTEGER                                    :: na, nb, nt

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

      call allocate_us_FR_pp_variables ()

      RETURN
      !
   end subroutine init_US_frpp_parameters
   !
   ! ==================================================================================
   subroutine allocate_US_frpp_variables ( )
      ! ---------------------------------------------------------------------------------

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
   end subroutine allocate_US_frpp_variables
   !
   ! =======================================================
   SUBROUTINE set_US_spin_orbit_operator ( frpp )
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

      !
      TYPE (pseudo_upf), intent(inout), TARGET   :: frpp (ntyp)

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
   END SUBROUTINE set_US_spin_orbit_operator
   !
   ! =======================================================================
   SUBROUTINE dvan_so_pauli_basis (frpp)
      ! --------------------------------------------------------------------
      !      transform dvan_so from |up,dw> basis to
      !      basis of pauli matrices
      !

      USE spin_orb,            ONLY : fcoef
      USE uspp_param,          ONLY : nh, nhm
      USE uspp,                ONLY : indv
      USE ions_base,           ONLY : nsp

      !
      IMPLICIT NONE

      !
      TYPE (pseudo_upf), intent(in), TARGET   :: frpp (nsp)

      !  internal variables

      integer                      :: ih, jh, vi, vj
      integer                      :: nt
      integer                      :: is1, is2, ijs
      INTEGER                      :: ierr

      !
      !  allocate Dso

      IF (.not. ALLOCATED (Vso_US) ) THEN
         ALLOCATE ( Vso_US (nhm,nhm,4,nsp), stat=ierr )
         if (ierr/=0) call errore ('dvan_so_pauli_basis', 'allocating Vso_US', abs (ierr))
      END IF
      !
      Vso_US = cmplx (0._dp, 0._dp)

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
                        Vso_US (ih,jh,ijs,nt) = frpp(nt)%dion(vi,vj) *    &
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
END MODULE spin_orbit_operator_uspp