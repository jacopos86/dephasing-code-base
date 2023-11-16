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
          D_so = (0.d0,0.d0)
          qq_so = (0.d0,0.d0)
          qq_at = 0.d0
          qq_nt = 0.d0
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
