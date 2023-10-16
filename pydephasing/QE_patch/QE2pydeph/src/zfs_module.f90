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
    ! --------------------------------------------------
    SUBROUTINE compute_rho12_G ( )
      ! ------------------------------------------------
      !
      !   compute :
      !           rho(G,-G) = 
      
      
      !  internal variables
      
      
      
      
      
      
      
      
      
      
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
  
