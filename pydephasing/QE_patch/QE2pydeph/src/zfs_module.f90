  !
  ! ----------------------------------------------------------------------------
  !
  !          zero field splitting tensor module
  !
  ! ----------------------------------------------------------------------------
  !
MODULE zfs_module
  !
  




  CONTAINS
    !
    SUBROUTINE allocate_array_variables ()
      
      
      
      
      
      !
      implicit none
      
      
      
      
      
      !
      !  allocate ZFS tensor
      
      
      
      
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
      
      
      
      
      !
      implicit none

      !    internal variables



      
      !
      !    allocate ddi_G
      !

      allocate ( ddi_G (ngm, 3, 3), stat=ierr )
      if (ierr/=0) call errore ('compute_ddig_space', 'allocating ddi_G', abs(ierr))
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
      
      !
      RETURN
      
      !        
    END SUBROUTINE compute_ddig_space
    
    
    
    
    
    
    !
END MODULE zfs_module
  
