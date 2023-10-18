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
    END SUBROUTINE allocate_array_variables
    
    
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
    SUBROUTINE compute_rho12_G ( ik, ib1, ib2, rhog )
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
      USE gvect,                 ONLY : ngm
      
      
      implicit none
      
      !  input variables

      integer, intent(in)             :: ik
      ! k pt.
      integer, intent(in)             :: ib1, ib2
      ! bands index
      complex(DP), intent(out)        :: rhog (1:ngm)
      ! rho(G,-G)
      
      !  internal variables
      real(DP), allocatable           :: gk (:)
      ! |G+k|^2
      complex(DP), allocatable        :: evc1_r (:,:)
      ! real space wfc1
      integer                         :: ir
      ! grid index
      integer                         :: ipol
      ! spin index
      integer                         :: ierr
      
      
      !
      !  plane waves
      !
      
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
      
      ! -----------------------------------------
      !     compute f1(r)
      ! -----------------------------------------

      !
      !  real space wfc
      !

      allocate ( evc1_r (1:dffts%nnr, 1:npol), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating evc1_r', ABS(ierr))

      !
      evc1_r (:,:) = cmplx (0._dp, 0._dp, kind=dp)
      do ig= 1, npw
         evc1_r (dffts%nl (igk_k(ig,ik)), 1) = evc (ig,ib1)
      end do

      !
      call invfft ('Wave', evc1_r (:,1), dffts)
      !
      IF (noncolin) THEN
         !
         do ig= 1, npw
            evc1_r (dffts%nl (igk_k(ig,ik)), 2) = evc (ig+npwx,ib1)
         end do
         !
         call invfft ('Wave', evc1_r (:,2), dffts)
         !
      END IF
      
      !
      allocate ( f1_aux (1:dffts%nnr), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_aux', ABS(ierr))
      !
      allocate ( f1_G (1:ngm, 1:npol), stat=ierr )
      if (ierr/=0) call errore ('compute_rho12_G','allocating f1_G', ABS(ierr))
      f1_G (:,:) = cmplx (0._dp, 0._dp, kind=dp)
      
      !
      !  run : ipol
      !
      
      DO ipol= 1, npol
         !
         f1_aux (:) = cmplx (0._dp, 0._dp, kind=dp)
         !
         do ir= 1, dffts%nnr
            f1_aux (ir) = evc1_r (ir,ipol) * conjg (evc1_r (ir,ipol))
         end do
         
         !
         !  compute f1(G)
         !
         
         call fwfft ('Rho', f1_aux, dffts)
         
         !
         f1_G (1:ngm, ipol) = f1_aux (dffts%nl (1:ngm))
         
      END DO
      
      WRITE(6,*) f1_G (1,:)






      
      
      
      
      
      
      
      
      !
    END SUBROUTINE compute_rho12_G
    
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
      
      
      
      
      implicit none
      
      !  internal variables
      
      
      
      
      
      
      
      
      
      
      
      
      
      

      
      
      
    END SUBROUTINE compute_zfs_tensor
    !
    
    
    
    
    !
END MODULE zfs_module
  
