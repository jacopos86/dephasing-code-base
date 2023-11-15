!
!   MODULE  :  spin orbit operator
!
MODULE spin_orbit_operator
  
  
  
  
  !
CONTAINS
  !
  ! =======================================================
  SUBROUTINE set_spin_orbit_operator ()
    ! -----------------------------------------------------
    
    USE uspp_param,       ONLY : upf, nh
    USE ions_base,        ONLY : ntyp => nsp
    USE io_global,        ONLY : stdout
    USE uspp,             ONLY : indv
    USE spin_orb,         ONLY : fcoef
    
    !
    implicit none
    
    !    internal variables
    
    integer                    :: nt, ih
    integer                    :: vi
    
    
    
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
       
       WRITE(stdout,*) "nbeta= ", upf(nt)%nbeta
       WRITE(stdout,*) upf(nt)%lll
       WRITE(stdout,*) upf(nt)%jjj
       !
    END DO
    
    
    
    
    
    
    
    
    
    
    
    
    
    RETURN
    !
  END SUBROUTINE set_spin_orbit_operator
  !
  
END MODULE spin_orbit_operator
