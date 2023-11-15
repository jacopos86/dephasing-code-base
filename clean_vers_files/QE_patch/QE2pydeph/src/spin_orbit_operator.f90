!
!   MODULE  :  spin orbit operator
!
MODULE spin_orbit_operator
  
  
  
  
  !
  
  SUBROUTINE set_spin_orbit_operator ()
    !
    
    
    !
    implicit none
    
    !    internal variables
    



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
