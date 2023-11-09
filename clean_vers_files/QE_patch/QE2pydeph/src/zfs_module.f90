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
  integer                       :: nmax
  !  n. occupied states
  !  max iter. index in sum
  
  
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
      USE lsda_mod,            ONLY : current_spin, lsda, isk
      USE klist,               ONLY : nks, wk
      USE io_global,           ONLY : stdout
      USE wvfct,               ONLY : wg, nbnd
      
      
      !    internal variables
      
      implicit none
      
      !
      integer                       :: ik, ib
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
      !  allocate transition table
      !
      
      allocate ( transitions_table (1:nmax*(nmax+1)/2, 1:4), stat=ierr )
      if (ierr/=0) call errore ('set_spin_band_index_occ_levels', 'allocating transitions', abs(ierr))
      transitions_table = 0
      
      !
      !  run ik
      !
      
      ij = 0
      !
      do ik1= 1, nks
         !
         !IF (lsda) current_spin  = isk(ik)
         !WRITE (stdout,*) "ik= ", ik, current_spin
         do ib1= 1, nbnd
            IF (ABS (occ (ik1,ib1) - 1.0) < eps4) THEN
               do ik2= 1, nks
                  do ib2= 1, nbnd
                     IF (ABS (occ (ik2,ib2) - 1.0) < eps4) THEN
                        ij = ij + 1
                        transitions_table (ij,1) = ik1
                        transitions_table (ij,2) = ib1
                        transitions_table (ij,3) = ik2
                        transitions_table (ij,4) = ib2
                     END IF
                  end do
               end do
            END IF
         end do
         !
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
    SUBROUTINE compute_Dab_ij ( ik )
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
      USE gvecw,                 ONLY : ecutwfc
      USE wvfct,                 ONLY : npwx
      USE io_files,              ONLY : iunwfc, nwordwfc
      USE io_global,             ONLY : stdout
      USE wavefunctions,         ONLY : evc
      USE cell_base,             ONLY : tpiba2
      USE klist,                 ONLY : ngk, xk, igk_k
      USE fft_interfaces,        ONLY : invfft, fwfft
      USE noncollin_module,      ONLY : npol
      USE cell_base,             ONLY : omega
      
      
      implicit none
      
      !  input variables

      integer, intent(in)             :: ik
      ! k pt.
      
      !  internal variables
      real(DP), allocatable           :: gk (:)
      ! |G+k|^2
      complex(DP), allocatable        :: evc1_r (:,:)
      ! real space wfc1
      complex(DP), allocatable        :: evc2_r (:,:)
      ! real space wfc2
      complex(DP), allocatable        :: f1_aux (:), f2_aux (:), f3_aux (:)
      ! aux. real space arrays
      complex(DP), allocatable        :: f1_G (:), f2_G (:), f3_G (:)
      ! aux. rec. space arrays
      complex(DP), allocatable        :: rhog (:)
      ! rho_12(G,-G)
      integer                         :: ir, ig
      ! grid index
      integer                         :: ipol
      ! spin index
      integer                         :: ib1, ib2, ij
      ! band index
      integer                         :: npw
      integer                         :: ierr
      
      
      !
      !  plane waves
      !
      WRITE (stdout,*) nmax
      ALLOCATE ( gk (1:npwx) )
      npw = ngk (ik)
      gk (:) = 0._DP

      !
      !  ecutwfc/tpiba2 = gcutw
      !

      call gk_sort ( xk (1,ik), ngm, g, ecutwfc/tpiba2, npw, igk_k (1,ik), gk )

      !
      !  read the wavefunction
      !
      
      call davcio (evc, 2*nwordwfc, iunwfc, ik, -1)

      !
      ! allocate arrays
      !
      
      allocate ( evc1_r (1:dffts%nnr, 1:npol), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating evc1_r', ABS(ierr))
      !
      allocate ( f1_aux (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_aux', ABS(ierr))
      !
      allocate ( f1_G (1:ngm), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_G', ABS(ierr))
      !
      allocate ( evc2_r (1:dffts%nnr, 1:npol), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating evc2_r', ABS(ierr))
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
      !  iterate ib1 : 1 -> nmax
      !
      
      ij = 1
      DO ib1= 1, nmax
         
         ! -----------------------------------------
         !     compute f1(r)
         ! -----------------------------------------
         
         !
         !  real space wfc
         !
         
         evc1_r (:,:) = cmplx (0._dp, 0._dp, kind=dp)
         do ig= 1, npw
            evc1_r (dffts%nl (igk_k(ig,ik)), 1) = evc (ig,ib1)
         end do
         
         !
         call invfft ('Wave', evc1_r (:,1), dffts)
         
         !
         f1_G (:) = cmplx (0._dp, 0._dp, kind=dp)
         
         !
         f1_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
         !
         do ir= 1, dffts%nnr
            f1_aux (ir) = evc1_r (ir,1) * conjg (evc1_r (ir,1))
         end do
         
         !
         !  compute f1(G)
         !
         
         call fwfft ('Rho', f1_aux, dffts)
         
         !
         f1_G (1:ngm) = f1_aux (dffts%nl (1:ngm))
         WRITE(6,*) f1_G (1), g (:,1), sum(evc (:,ib1) * conjg(evc(:,ib1)))
         call stop_pp
         
         !
         !  run over ib2 : 1 -> ib1
         !
         
         DO ib2= 1, ib1

            ! ----------------------------------------
            !   compute f2(r)
            ! ----------------------------------------
            
            !
            !  real space evc2_r
            !
            
            evc2_r (:,:) = cmplx (0._dp, 0._dp) 
            !
            do ig= 1, npw
               evc2_r (dffts%nl (igk_k(ig,ik)), 1) = evc (ig,ib2)
            end do
            
            !
            call invfft ('Wave', evc2_r (:,1), dffts)
            !
            
            f2_G (:) = cmplx (0._dp, 0._dp, kind=dp)
            
            !
            f2_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
            !
            do ir= 1, dffts%nnr
               f2_aux (ir) = evc2_r (ir,1) * conjg (evc2_r (ir,1))
            end do
            
            !
            !  compute f2(G)
            !
            
            call fwfft ('Rho', f2_aux, dffts)
            
            !
            f2_G (1:ngm) = f2_aux (dffts%nl (1:ngm))
            
            WRITE(stdout,*) f2_G (1)
            
            !
            !  compute -> f3(r) = psi1(r)* psi2(r)
            !
            
            f3_G (:) = cmplx (0._dp, 0._dp, kind=dp)
            
            !
            f3_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
            !
            do ir= 1, dffts%nnr
               f3_aux (ir) = conjg (evc1_r (ir,1)) * evc2_r (ir,1)
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
            
            !
            ij = ij + 1
            !
         END DO
         
         !
      END DO
      
      !   ADJUST UNITS
      Dab_ij (:,:,:) = Dab_ij (:,:,:) * omega
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
      
      USE noncollin_module,        ONLY : npol
      
      
      implicit none
      
      !  internal variables
      
      
      
      !
      IF (npol > 1) call errore ('compute_zfs_tensor','calculation must be collinear', 1)
      
      !
      !  compute Dab(i,j)
      !
      
      call compute_Dab_ij ( 1 )
      
      
      
      
      
      
      
      

      
      
      
    END SUBROUTINE compute_zfs_tensor
    !
    
    
    
    
    !
END MODULE zfs_module
  
