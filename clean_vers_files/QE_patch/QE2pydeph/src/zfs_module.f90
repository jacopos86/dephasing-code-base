!
! ----------------------------------------------------------------------------
!
!          zero field splitting tensor module
!
! ----------------------------------------------------------------------------
!
MODULE zfs_module
   !

   USE kinds,               ONLY : DP
  
   real(DP), allocatable         :: ddi_G (:,:,:)
   !
   !  dip. dip. inter. (G)
   real(DP), allocatable         :: ddi_r (:,:,:)
   !
   !  dip. dip. inter. (r)
   complex(DP), allocatable      :: Dab_ij (:,:,:)
   !
   !  Dab(i,j)
   real(DP)                      :: Dab (3,3)
   !  D tensor
   real(DP)                      :: Dso_ab (3,3)
   !  SOC D tensor
   real(DP)                      :: D, E
   !  D coefficients
   integer, allocatable          :: transitions_table (:,:)
   !  transitions index
   integer                       :: nmax
   !  n. occupied states
   !  max iter. index in sum
   integer                       :: niter
   !  tot. number iterations
   !  to execute
   
   CONTAINS
   !
   SUBROUTINE allocate_zfs_array_variables ()
    
      USE gvect,      ONLY : ngm
      USE fft_base,   ONLY : dfftp
      
      !
      implicit none
      
      !
      integer             :: ierr
      
      !
      !    allocate ddi_G
      !
      
      allocate ( ddi_G (ngm, 3, 3), stat=ierr )
      if (ierr/=0) call errore ('allocate_array_variables', 'allocating ddi_G', abs(ierr))
      
      !
      !    allocate ddi_r
      !
      
      allocate ( ddi_r (dfftp%nnr, 3, 3), stat=ierr )
      if (ierr/=0) call errore ('allocate_array_variables', 'allocating ddi_r', abs(ierr))
      
      !
      !    allocate Dab(i,j)
      !
      
      allocate ( Dab_ij (1:nmax*(nmax+1)/2, 1:3, 1:3), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating Dab_ij', ABS(ierr))
      Dab_ij = cmplx (0._dp, 0._dp)
      
      !
   END SUBROUTINE allocate_zfs_array_variables

   ! ================================================================
   SUBROUTINE set_spin_band_index_occ_levels ( )
      ! ==============================================================

      USE constants,           ONLY : eps4
      USE lsda_mod,            ONLY : lsda, isk
      USE klist,               ONLY : nks, wk
      USE wvfct,               ONLY : wg, nbnd
      USE io_global,           ONLY : stdout
      
      
      !    internal variables
      
      implicit none
      
      !
      integer                       :: ik, ib, ij, i, ib_i, ib_j, ik_i, ik_j, isp_i, isp_j, j
      INTEGER                       :: ierr
      !
      integer, allocatable          :: en_level_index (:,:)
      real(DP)                      :: occ (nks,nbnd)
      
      
      !  compute occup.
      
      occ = 0._dp
      nmax = 0
      
      do ik= 1, nks
         !
         do ib=1, nbnd
            occ (ik,ib) = wg (ib,ik) / wk (ik)
            IF (ABS (occ (ik,ib) - 1.0) < eps4 ) nmax = nmax + 1
         end do
         !
      end do
      
      !
      allocate (en_level_index (1:nmax,1:2), stat=ierr )
      if (ierr/=0) call errore ('set_spin_band_index_occ_levels', 'allocating en_level_index', abs(ierr))
      en_level_index = 0
      i = 0
      !
      do ik= 1, nks
         do ib= 1, nbnd
            IF (ABS (occ (ik,ib) - 1.0) < eps4 ) THEN
               i = i + 1
               en_level_index (i,1) = ik
               en_level_index (i,2) = ib
            END IF
         end do
      end do
      
      !
      !  allocate transition table
      !
      
      allocate ( transitions_table (1:nmax*(nmax+1)/2, 1:6), stat=ierr )
      if (ierr/=0) call errore ('set_spin_band_index_occ_levels', 'allocating transitions', abs(ierr))
      transitions_table = 0
      niter = nmax*(nmax+1)/2
      
      !
      !  run ik
      !
      
      ij = 0
      !
      do i= 1, nmax
         !
         ik_i = en_level_index (i,1)
         ib_i = en_level_index (i,2)
         IF (lsda) isp_i = isk(ik_i)
         !
         do j= i, nmax
            !
            ik_j = en_level_index (j,1)
            ib_j = en_level_index (j,2)
            IF (lsda) isp_j = isk(ik_j)
            !
            ij = ij + 1
            transitions_table (ij,1) = ik_i
            transitions_table (ij,2) = ib_i
            transitions_table (ij,3) = isp_i
            transitions_table (ij,4) = ik_j
            transitions_table (ij,5) = ib_j
            transitions_table (ij,6) = isp_j
            !WRITE(stdout,*) ij, transitions_table (ij,:)
         end do
      end do
      !
      
      RETURN
      !
   END SUBROUTINE set_spin_band_index_occ_levels
   !
   ! ================================================================
   SUBROUTINE set_SOC_transitions_list ( transitions_list, ntr )
      ! ==============================================================
      
      USE constants,                  ONLY : eps4
      USE lsda_mod,                   ONLY : lsda, isk
      USE klist,                      ONLY : nks, wk
      USE wvfct,                      ONLY : wg, nbnd
      USE io_global,                  ONLY : stdout
      
      implicit none

      integer, intent(out), allocatable   :: transitions_list (:,:)
      integer, intent(out)                :: ntr
      
      !    internal variables
      !
      integer                             :: ik, ik_i, ik_j, ib, ib_i, ib_j, isp_i, isp_j, io, ni, itr
      integer                             :: nocc, nunocc
      integer, allocatable                :: occ_states (:,:), unocc_states (:,:)
      INTEGER                             :: ierr
      !
      real(DP)                            :: occ (nks,nbnd)
      
      
      !  compute occup.
      
      occ = 0._dp
      nocc = 0
      nunocc = 0
      
      do ik= 1, nks
         !
         do ib=1, nbnd
            occ (ik,ib) = wg (ib,ik) / wk (ik)
            IF (ABS (occ (ik,ib) - 1.0) < eps4 ) THEN
               nocc = nocc + 1
            ELSE
               nunocc = nunocc + 1
            END IF
         end do
         !
      end do

      ! number of transitions

      ntr = nocc * nunocc
      io = 0
      ni = 0
      allocate ( occ_states (nocc,2), unocc_states (nunocc,2) )
      occ_states = 0
      unocc_states = 0
      
      !
      do ik= 1, nks
         !
         do ib= 1, nbnd
            IF (ABS (occ (ik,ib) - 1.0) < eps4 ) THEN
               io = io + 1
               occ_states (io,1) = ik
               occ_states (io,2) = ib
            ELSE
               ni = ni + 1
               unocc_states (ni,1) = ik
               unocc_states (ni,2) = ib
            END IF
         end do
         !
      end do
      
      !
      !  allocate transition table
      !
      
      allocate ( transitions_list (1:ntr, 1:6), stat=ierr )
      if (ierr/=0) call errore ('set_spin_band_index_occ_levels', 'allocating transitions_list', abs(ierr))
      transitions_list = 0
      
      !
      !  run ik
      !
      
      itr = 0
      !
      do io= 1, nocc
         !
         ik_i = occ_states (io,1)
         ib_i = occ_states (io,2)
         IF (lsda) isp_i = isk(ik_i)
         !
         do ni= 1, nunocc
            !
            ik_j = unocc_states (ni,1)
            ib_j = unocc_states (ni,2)
            IF (lsda) isp_j = isk(ik_j)
            !
            itr = itr + 1
            transitions_list (itr,1) = ik_i
            transitions_list (itr,2) = ib_i
            transitions_list (itr,3) = isp_i
            transitions_list (itr,4) = ik_j
            transitions_list (itr,5) = ib_j
            transitions_list (itr,6) = isp_j
         end do
      end do
      !
      RETURN
      !
   END SUBROUTINE set_SOC_transitions_list
   !
   ! ================================================================
   SUBROUTINE compute_ddig_space ( )
      ! ==============================================================
      !
      !    compute dipole - dipole interaction in G space
      !
      !    ddi(G)_{xy} = 4 * pi * e^2 * [Gx * Gy / G^2 - delta(x,y) / 3 ]
      !
      
      USE constants,    ONLY : eps8, fpi, e2
      USE cell_base,    ONLY : omega
      USE gvect,        ONLY : ngm, g
      USE io_global,    ONLY : stdout
      
      !
      implicit none
      
      !    internal variables
      !
      integer               :: x, y
      integer               :: ng
      !
      real(DP)              :: gsq
      
      !    initialize ddi (G)
      
      ddi_G = 0._dp
      
      !    iterate over G vectors
      
      do ng= 1, ngm
         gsq = SUM ( g (:,ng) ** 2 )
         !  no G=0 term
         IF (gsq > eps8) THEN
            do x= 1, 3
               do y= 1, x
                  if (x == y) then
                     ddi_G (ng,x,y) = g (x,ng) * g (y,ng) / gsq - 1._dp / 3
                  else
                     ddi_G (ng,x,y) = g (x,ng) * g (y,ng) / gsq
                     ddi_G (ng,y,x) = ddi_G (ng,x,y)
                  end if
               end do
            end do
         ENDIF
      end do
      !
      ddi_G (:,:,:) = ddi_G (:,:,:) * fpi * e2 / omega
      !  bohr^-3
      
      !
      RETURN
      
      !        
   END SUBROUTINE compute_ddig_space
    
   ! -----------------------------------------------------------------
   !  compute ddi (r) : inv fft
   !
   ! ----------------------------------------------------------------
   SUBROUTINE compute_invfft_ddiG ()
      ! -------------------------------------------------------------

      USE constants,                ONLY : eps8
      USE mp,                       ONLY : mp_sum
      USE fft_base,                 ONLY : dfftp
      USE mp_bands,                 ONLY : intra_bgrp_comm
      USE control_flags,            ONLY : gamma_only
      USE gvect,                    ONLY : ngm, gg
      USE fft_interfaces,           ONLY : invfft
      
      !
      implicit none
      
      !   internal variables

      integer                           :: x, y
      !   x,y counters
      integer                           :: ng
      integer                           :: ierr
      !
      complex(DP), allocatable          :: aux_arr (:)
      !
      real(DP)                          :: ddi_of_0
      
      
      !
      !  allocate temp. array
      
      allocate ( aux_arr (dfftp%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_invfft_ddiG', 'allocating aux_arr', abs(ierr))
      
      !
      !  prepare array for invfft
      !
      
      DO x= 1, 3
         DO y= x, 3
            
            aux_arr (:) = (0._dp,0._dp)
            !
            DO ng= 1, ngm
               aux_arr (dfftp%nl (ng)) = aux_arr (dfftp%nl (ng)) + ddi_G (ng,x,y)
            END DO

            !
            IF (gamma_only) THEN
               DO ng= 1, ngm
                  aux_arr (dfftp%nlm (ng)) = CONJG ( aux_arr (dfftp%nl (ng)))
               END DO
            END IF

            !
            !  ... ddi_of_0  ddi(G=0)
            !

            ddi_of_0 = 0.0_DP
            IF (gg(1) < eps8) ddi_of_0 = DBLE( aux_arr (dfftp%nl(1)) )

            !
            call mp_sum ( ddi_of_0, intra_bgrp_comm )

            !
            ! inv FFT to real space
            !

            call invfft ('Rho', aux_arr, dfftp)

            !
            IF (x == y) THEN
               ddi_r (:,x,y) = DBLE( aux_arr (:) )
            ELSE
               ddi_r (:,x,y) = DBLE( aux_arr (:) )
               ddi_r (:,y,x) = DBLE( aux_arr (:) )
            END IF
            !

         END DO
         !
      END DO
      
      !
   END SUBROUTINE compute_invfft_ddiG

   !
   ! -------------------------------------------------------------
   SUBROUTINE compute_Dab_ij ( )
      ! ----------------------------------------------------------
      !
      !   Compute rho(G, -G) for two electrons.
      !
      !   rho(G, -G) = f1(G) * f2(-G) - |f3(G)|^2,
      !   = f1(G) * conj(f2(G)) - f3(G) * conj(f3(G))
      !   f1, f2 and f3 are given in PRB 77, 035119 (2008):
      !   f1(r) = |psi1(r)|^2
      !   f2(r) = |psi2(r)|^2
      !   f3(r) = conj(psi1(r)) * psi2(r)
      !   f1(G), f2(G) and f3(G) are computed by Fourier Transform
      !   of f1(r), f2(r) and f3(r)
      
      USE fft_base,              ONLY : dffts
      USE io_global,             ONLY : stdout
      USE fft_interfaces,        ONLY : fwfft
      USE noncollin_module,      ONLY : npol
      USE cell_base,             ONLY : omega
      USE physical_constants,    ONLY : D0
      USE control_flags,         ONLY : gamma_only
      USE wavefunctions,         ONLY : psic, evc
      USE klist,                 ONLY : ngk, igk_k
      USE gvect,                 ONLY : gstart, ngm
      
      implicit none
      
      !  internal variables
      !
      complex(DP), allocatable        :: f1_G (:,:), f2_G (:,:), f3_G (:,:)
      ! aux. rec. space arrays
      complex(DP), allocatable        :: evci_r (:), evcj_r (:)
      ! aux. wave functions
      complex(DP), allocatable        :: rhog (:,:)
      ! rho_12(G,-G)
      integer                         :: ij, ib_i, ib_j, ik_i, ik_j, ik_ij, ir, ig
      ! indexes
      integer                         :: npw_i, npw_ij, npw_j
      ! n. pws
      integer                         :: ierr
      !
      
      !
      !  allocate real space wfc
      
      allocate ( evci_r (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating evci_r', ABS(ierr))
      
      !
      allocate ( evcj_r (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating evcj_r', ABS(ierr))
      
      !
      ! allocate arrays
      !
      
      allocate ( f1_G (ngm,2), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_G', ABS(ierr))
      !
      allocate ( f2_G (ngm,2), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f2_G', ABS(ierr))
      !
      allocate ( f3_G (ngm,2), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f3_G', ABS(ierr))
      !
      allocate ( rhog (ngm,2), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating rhog', ABS(ierr))
      
      !
      !  iterate ij : 1 -> niter
      !
      
      DO ij= 1, niter
         !
         
         ik_i = transitions_table (ij,1)
         ib_i = transitions_table (ij,2)
         
         ! -----------------------------------------
         !     compute f1(r)
         ! -----------------------------------------
         
         evci_r (:) = cmplx (0._dp, 0._dp)
         !
         IF (gamma_only) THEN
            call extract_evc_r_gamma_only ( ik_i, ib_i, evci_r )
         ELSE
            call extract_evc_r_ofk ( ik_i, ib_i, evci_r )
         END IF
         
         !
         f1_G (:,:) = cmplx (0._dp, 0._dp, kind=dp)
         psic (:) = (0.d0, 0.d0)
         
         !
         do ir= 1, dffts%nnr
            psic (ir) = evci_r (ir) * CONJG (evci_r (ir))
         end do
         
         !
         !  compute f1(G)
         !
         
         call fwfft ('Wave', psic, dffts)
         
         !
         npw_i = ngk (ik_i)
         f1_G (1:npw_i,1) = psic (dffts%nl (igk_k (1:npw_i,ik_i)))
         !
         IF (gamma_only) THEN
            IF (gstart==2) psic (dffts%nlm(1)) = (0.d0,0.d0)
            f1_G (1:npw_i,2) = psic(dffts%nlm(igk_k (1:npw_i,ik_i)))
            ! -G index
         END IF
         
         !WRITE(6,*) f1_G (1,:), g (:,1), sum(evc (:,ib_i) * conjg(evc(:,ib_i)))
         
         ! ----------------------------------------
         !   compute f2(r)
         ! ----------------------------------------
         
         ik_j = transitions_table (ij,4)
         ib_j = transitions_table (ij,5)
         
         !
         !  real space evc2_r
         !
         
         evcj_r (:) = cmplx (0._dp, 0._dp) 
         
         !
         IF (gamma_only) THEN
            call extract_evc_r_gamma_only ( ik_j, ib_j, evcj_r )
         ELSE
            call extract_evc_r_ofk ( ik_j, ib_j, evcj_r )
         END IF
         
         !
         f2_G (:,:) = cmplx (0._dp, 0._dp, kind=dp)
         psic (:) = (0.d0,0.d0)
         
         !
         do ir= 1, dffts%nnr
            psic (ir) = evcj_r (ir) * CONJG (evcj_r (ir))
         end do
         
         !
         !  compute f2(G)
         !
         
         call fwfft ('Wave', psic, dffts)
         
         !
         npw_j = ngk (ik_j)
         f2_G (1:npw_j,1) = psic (dffts%nl (igk_k (1:npw_j,ik_j)))
         
         !
         IF (gamma_only) THEN
            IF (gstart==2) psic (dffts%nlm(1)) = (0.d0,0.d0)
            f2_G (1:npw_j,2) = psic (dffts%nlm(igk_k (1:npw_j,ik_j)))
            ! -G index
         END IF
         
         !WRITE(6,*) f2_G (1,:), g (:,1), sum(evc (:,ib_j) * conjg(evc(:,ib_j)))
         
         ! =========================================================
         !  compute -> f3(r) = psi1(r)* psi2(r)
         ! =========================================================
         
         f3_G (:,:) = cmplx (0._dp, 0._dp, kind=dp)   
         psic (:) = (0.d0,0.d0)
         
         !
         do ir= 1, dffts%nnr
            psic (ir) = CONJG (evci_r (ir)) * evcj_r (ir)
         end do
         
         !
         !  compute f3 (G)
         !
         
         call fwfft ('Wave', psic, dffts)
         
         !
         IF (npw_i < npw_j ) THEN
            ik_ij = ik_i
            npw_ij = npw_i
         ELSE
            ik_ij = ik_j
            npw_ij = npw_j
         END IF
         !
         f3_G (1:npw_ij,1) = psic (dffts%nl (igk_k (1:npw_ij,ik_ij)))
         
         !
         IF (gamma_only) THEN
            IF (gstart==2) psic (dffts%nlm(1)) = (0.d0,0.d0)
            f3_G (1:npw_ij,2) = psic (dffts%nlm(igk_k (1:npw_ij,ik_ij)))
            ! -G index
         END IF
         
         !
         !WRITE(stdout,*) f3_G (1,:), g (:,1), sum(evc (:,ib_i) * conjg(evc(:,ib_j)))
         
         ! --------------------------------------------------------
         !  comoute -> rho(G)
         ! --------------------------------------------------------
         
         rhog (:,:) = cmplx (0._dp, 0._dp, kind=dp)
         rhog (:,:) = f1_G (:,:) * conjg (f2_G (:,:)) - f3_G (:,:) * conjg (f3_G (:,:))
         
         !
         !   compute Dab_ij
         !
         
         do ig= 1, ngm
            Dab_ij (ij,:,:) = Dab_ij (ij,:,:) + rhog (ig,1) * ddi_G (ig,:,:)
         end do
         !
         IF (gamma_only) THEN
            do ig= 1, ngm
               Dab_ij (ij,:,:) = Dab_ij (ij,:,:) + rhog (ig,2) * ddi_G (ig,:,:)
            end do
         END IF
         
         !
      END DO
      
      !   ADJUST UNITS
      Dab_ij (:,:,:) = Dab_ij (:,:,:) * D0
      ! bohr ^ -3
      
      !
   END SUBROUTINE compute_Dab_ij
    
   !
   ! =====================================================
   SUBROUTINE set_zfs_tensor ( )
      ! ---------------------------------------------------
      !
      !   This subroutine compute the ZFS tensor
      !   D =
      !
      !   in presence of spin orbit
      !   -> correction
      !
      !
      
      USE constants,               ONLY : eps4
      USE noncollin_module,        ONLY : npol
      USE io_global,               ONLY : stdout
      USE physical_constants,      ONLY : compute_prefactor
      
      implicit none
      
      !  internal variables
      
      real(DP)                         :: chi_ij
      real(DP)                         :: A (3,3)
      real(DP)                         :: EIG (3), W (3)
      real(DP)                         :: Dx, Dy, Dz
      real(DP), allocatable            :: WORK (:)
      !
      integer                          :: ij, i, j
      integer                          :: isp_i, isp_j
      integer                          :: INFO, LDA, LWORK
      integer                          :: INDX (3)
      integer                          :: N = 3
      
      !
      IF (npol > 1) call errore ('compute_zfs_tensor','calculation must be collinear', 1)
      
      call compute_prefactor ( )
      
      !
      !  compute Dab(i,j)
      !
      
      call compute_Dab_ij ( )
      
      !
      !  compute Dab tensor  ->  Dab = \sum_{i<j} Dab(i,j) chi(i,j)
      !
      
      Dab = 0._dp
      !
      do ij= 1, niter
         
         !
         isp_i = transitions_table (ij,3)
         isp_j = transitions_table (ij,6)
         !
         if (isp_i == isp_j) THEN
            chi_ij = 1._dp
         else
            chi_ij =-1._dp
         end if
         
         !
         Dab (:,:) = Dab (:,:) + chi_ij * Dab_ij (ij,:,:)
         !
      end do
      !
      WRITE(stdout,*) Dab (1,1), Dab (1,2), Dab (1,3)
      WRITE(stdout,*) Dab (2,1), Dab (2,2), Dab (2,3)
      WRITE(stdout,*) Dab (3,1), Dab (3,2), Dab (3,3)
      
      !  SYMMETRIZE D matrix
      !
      
      A = 0._dp
      do i= 1, 3
         do j= i, 3
            A (i,j) = Dab (i,j)
            A (j,i) = Dab (i,j)
         end do
      end do
      !
      
      WRITE(stdout,*) "Tr {Dab} = ", Dab(1,1)+Dab(2,2)+Dab(3,3)
      
      !
      !  compute eigenvectors/eigenvalues
      
      W = 0._dp
      LDA = max (1, N)
      LWORK = max (1, 3*N-1)
      allocate ( WORK (1:max (1, LWORK)) )
      WORK = 0._dp
      
      !
      call DSYEV ('V', 'U', N, A, LDA, W, WORK, LWORK, INFO )
      
      !
      IF (INFO == 0) THEN
         EIG (:) = ABS (W (:))
         INDX = 0
         call hpsort_eps (N, EIG, INDX, eps4)
         
         !
         Dz = W (INDX (N))
         D  = 1.5 * Dz
         !
         Dy = W (INDX(1))
         Dx = W (INDX(2))
         E  = 0.5 * (Dx - Dy)
         !
      ELSE
         !
         call errore ('error in compute_zfs_tensor : DSYEV', INFO)
      END IF
      !
      WRITE(stdout,*) "D= ", D
      WRITE(stdout,*) "E= ", E
      
      RETURN
      !
    END SUBROUTINE set_zfs_tensor
    !
    ! =========================================================================
    SUBROUTINE compute_zfs_tensor ()
      ! -----------------------------------------------------------------------
      
      implicit none
      
      !
      !  set nmax and arrays
      
      call set_spin_band_index_occ_levels ()
      
      !
      call allocate_zfs_array_variables ()
      
      !
      !  compute ddi (G)
      !
      
      call compute_ddig_space ()
      
      !
      !  compute ddi real space
      !
      
      call compute_invfft_ddiG ()
      
      !
      !  compute Dab
      !
      
      call set_zfs_tensor ( )
      
      !
      RETURN
      !
   END SUBROUTINE compute_zfs_tensor
    
   ! ==========================================================================
   SUBROUTINE compute_soc_zfs_tensor ()
      ! -----------------------------------------------------------------------

      USE bec_module,               ONLY : allocate_bec_arrays, compute_bec_array
      
      implicit none

      !
      integer                           :: ntr
      integer, allocatable              :: transitions_list (:,:)
      
      !
      call set_SOC_transitions_list ( transitions_list, ntr )
      
      !
      !   compute <beta|Psi>
      !
      
      call allocate_bec_arrays ()
      
      !
      call compute_bec_array ()
      
      !
      !   compute <Psi_o|Hso|Psi_n>
      !
      
      call compute_soc_matrix_elements ( transitions_list, ntr )
      
      !
      call set_soc_zfs_tensor ( transitions_list, ntr )
      
      !
      RETURN
      !
   END SUBROUTINE compute_soc_zfs_tensor
    
   ! ==========================================================================
   SUBROUTINE set_soc_zfs_tensor ( transitions_list, ntr )
      ! ------------------------------------------------------------------------
      !
      !      ESO_ab = \sum_o,s,s' \sum_n^unocc Re{ <Psi_o^s|HSO^a|Psi_n^s'>
      !                  <Psi_n^s'|HSO^b|Psi_o^s> / (e_o(s) - e_n(s'))
      
      USE constants,                    ONLY : ELECTRONVOLT_SI, RYTOEV
      USE physical_constants,           ONLY : Hz_to_joule
      USE wvfct,                        ONLY : et
      USE io_global,                    ONLY : stdout
      USE spin_operators,               ONLY : HSO_a
      
      !
      IMPLICIT NONE
      
      integer, intent(in)                    :: ntr
      integer, intent(in)                    :: transitions_list (ntr,6)
      
      !     internal variables
      
      integer                                :: a, b, itr
      integer                                :: ki, kpi, ni, oi, si, spi
      
      !
      !     iterate over transitions
      !
      WRITE(stdout,*) ntr
      do itr= 1, ntr
         !
         ki = transitions_list (itr, 1)
         oi = transitions_list (itr, 2)
         si = transitions_list (itr, 3) 
         kpi= transitions_list (itr, 4)
         ni = transitions_list (itr, 5)
         spi= transitions_list (itr, 6)
         !
         do a= 1, 3
            do b= 1, 3
               ! TODO
               ! correct here Dso_ab iteration
               DSO_ab (a,b) = DSO_ab (a,b) +        &
                    real (HSO_a (itr,a) * conjg (HSO_a (itr,b))) / (et (oi,ki) - et (ni,kpi))
               !
               !  Ry units
               !
            end do
         end do
         !
      end do
      !
      !  conversion to MHz
      DSO_ab (:,:) = DSO_ab (:,:) * RYTOEV * ELECTRONVOLT_SI
      !  Joule units
      DSO_ab (:,:) = DSO_ab (:,:) / Hz_to_joule * 1.e-6
      !  MHz units
      WRITE(stdout,*) DSO_ab (1,1), DSO_ab (2,2), DSO_ab (3,3), DSO_ab (1,2)
      
      !
      RETURN
      !
   END SUBROUTINE set_soc_zfs_tensor
   !
END MODULE zfs_module