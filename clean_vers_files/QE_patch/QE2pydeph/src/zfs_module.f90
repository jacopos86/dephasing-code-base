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
    SUBROUTINE allocate_array_variables ()
      
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
    END SUBROUTINE allocate_array_variables

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
    SUBROUTINE compute_ddig_space ( )
      ! ==============================================================
      !
      !    compute dipole - dipole interaction in G space
      !
      !    ddi(G)_{xy} = 4 * pi * [Gx * Gy / G^2 - delta(x,y) / 3 ]
      !
      
      USE constants,    ONLY : eps8, fpi
      USE cell_base,    ONLY : omega
      USE gvect,        ONLY : ngm, g
      USE io_global,    ONLY : stdout
      
      
      !
      implicit none
      
      !    internal variables
      
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
      !WRITE(6,*) ddi_G (:,1,2)
      !
      ddi_G (:,:,:) = ddi_G (:,:,:) * fpi / omega
      !  bohr^-3
      
      !
      RETURN
      
      !        
    END SUBROUTINE compute_ddig_space

    !
    !  compute ddi (r) : inv fft
    !

    ! ----------------------------------------------------------------
    SUBROUTINE compute_invfft_ddiG ()
      ! --------------------------------------------------------------

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
    ! ------------------------------------------------------------
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
      USE gvect,                 ONLY : ngm, g
      USE io_global,             ONLY : stdout
      USE fft_interfaces,        ONLY : fwfft
      USE wavefunctions,         ONLY : evc
      USE noncollin_module,      ONLY : npol
      USE cell_base,             ONLY : omega
      USE klist,                 ONLY : nks
      USE wvfct,                 ONLY : nbnd
      USE physical_constants,    ONLY : D0
      USE control_flags,         ONLY : gamma_only
      
      USE cell_base,             ONLY : tpiba2
  USE io_files,              ONLY : iunwfc, nwordwfc
  USE wavefunctions,         ONLY : evc
  USE klist,                 ONLY : nks, ngk, xk, igk_k
  USE wvfct,                 ONLY : nbnd, npwx
  USE gvect,                 ONLY : g, ngm
  USE gvecw,                 ONLY : ecutwfc
      implicit none
      
      !  internal variables
      complex(DP), allocatable        :: f1_aux (:), f2_aux (:), f3_aux (:)
      ! aux. real space arrays
      complex(DP), allocatable        :: f1_G (:), f2_G (:), f3_G (:)
      ! aux. rec. space arrays
      complex(DP), allocatable        :: evc_r (:,:,:)
      ! real space wfc
      complex(DP), allocatable        :: rhog (:)
      ! rho_12(G,-G)
      integer                         :: ij, ib_i, ib_j, ig, ik, ik_i, ik_j, ir
      ! band index
      integer                         :: ierr
      !
      logical                         :: use_tg
      
      integer                         :: npw
  real(DP), allocatable           :: gk (:)
      
      !
      !  allocate real space wfc
      
      allocate ( evc_r (1:dffts%nnr, 1:nks, 1:nbnd), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating evc_r', ABS(ierr))
      evc_r = cmplx (0._dp, 0._dp)

      !
      use_tg = dffts%has_task_groups
      
      !
      !  produce real space wave functions
      !
      
      do ik= 1, 1
         !
         call extract_real_space_wfc ( ik, evc_r )
         !
      end do
      
      !
      ! allocate arrays
      !
      
      allocate ( f1_aux (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_aux', ABS(ierr))
      !
      allocate ( f1_G (1:ngm), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_G', ABS(ierr))
      !
      allocate ( f2_aux (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f2_aux', ABS(ierr))
      !
      allocate ( f2_G (1:ngm), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f2_G', ABS(ierr))
      !
      allocate ( f3_aux (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f3_aux', ABS(ierr))
      !
      allocate ( f3_G (1:ngm), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f3_G', ABS(ierr))
      !
      allocate ( rhog (1:ngm), stat=ierr )
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
         
         !
         !evci_r (:,:) = cmplx (0._dp, 0._dp, kind=dp)
         !do ig= 1, npw
         !   evci_r (dffts%nl (igk_k(ig,ik_i)), 1) = evc (ig,ib_i)
         !end do
         
         !
         !call invfft ('Wave', evc1_r (:,1), dffts)
         
         !
         f1_G (:) = cmplx (0._dp, 0._dp, kind=dp)
         !
         f1_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
         
         !
         do ir= 1, dffts%nnr
            f1_aux (ir) = evc_r (ir,ik_i,ib_i) * conjg (evc_r (ir,ik_i,ib_i))
         end do
         
         !
         !  compute f1(G)
         !
         
         call fwfft ('Rho', f1_aux, dffts)
         
         !
         f1_G (1:ngm) = f1_aux (dffts%nl (1:ngm))
         
         !WRITE(6,*) f1_G (1), g (:,1), sum(evc (:,ib_i) * conjg(evc(:,ib_i)))
         
         ! ----------------------------------------
         !   compute f2(r)
         ! ----------------------------------------
         
         ik_j = transitions_table (ij,4)
         ib_j = transitions_table (ij,5)
         
         !
         !  real space evc2_r
         !
         
         !evc2_r (:,:) = cmplx (0._dp, 0._dp) 
         !
         !do ig= 1, npw
         !evc2_r (dffts%nl (igk_k(ig,ik)), 1) = evc (ig,ib2)
         !end do
         
         !
         !call invfft ('Wave', evc2_r (:,1), dffts)
         !
         
         f2_G (:) = cmplx (0._dp, 0._dp, kind=dp)
         !
         f2_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
         !
         do ir= 1, dffts%nnr
            f2_aux (ir) = evc_r (ir,ik_j,ib_j) * conjg (evc_r (ir,ik_j,ib_j))
         end do
         
         !
         !  compute f2(G)
         !
         
         call fwfft ('Rho', f2_aux, dffts)
         
         !
         f2_G (1:ngm) = f2_aux (dffts%nl (1:ngm))
         
         ! =========================================================
         !  compute -> f3(r) = psi1(r)* psi2(r)
         ! =========================================================
         
         f3_G (:) = cmplx (0._dp, 0._dp, kind=dp)   
         !
         f3_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
         !
         do ir= 1, dffts%nnr
            f3_aux (ir) = conjg (evc_r (ir,ik_i,ib_i)) * evc_r (ir,ik_j,ib_j)
         end do
         
         !
         !  compute f3 (G)
         !
         
         call fwfft ('Rho', f3_aux, dffts)
         
         !
         f3_G (1:ngm) = f3_aux (dffts%nl (1:ngm))
         
         !
         rhog (:) = cmplx (0._dp, 0._dp, kind=dp)
         rhog (:) = f1_G (:) * conjg (f2_G (:)) - f3_G (:) * conjg (f3_G (:))
         
         !
         !   compute Dab_ij
         !
         
         do ig= 1, ngm
            Dab_ij (ij,:,:) = Dab_ij (ij,:,:) + rhog (ig) * ddi_G (ig,:,:)
         end do
         WRITE(stdout,*) ik_i, ib_i, ik_j, ib_j, Dab_ij (ij,1,2)
         
         !
      END DO
      
      !   ADJUST UNITS
      Dab_ij (:,:,:) = Dab_ij (:,:,:) * D0
      ! bohr ^ -3
      
      !
    END SUBROUTINE compute_Dab_ij
    
    !
    ! --------------------------------------------------
    SUBROUTINE compute_zfs_tensor ( )
      ! ------------------------------------------------
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
      WRITE(stdout,*) Dab
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
      WRITE(stdout,*) D, E
      
      RETURN
      !
    END SUBROUTINE compute_zfs_tensor
    !

    ! ==============================================================
    SUBROUTINE compute_soc_zfs_tensor ()
      ! ------------------------------------------------------------
      
      
      
      IMPLICIT NONE
      
      !     internal variables
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

      
      
      
      
      
      
      
      
    END SUBROUTINE compute_soc_zfs_tensor
    !
END MODULE zfs_module
  
