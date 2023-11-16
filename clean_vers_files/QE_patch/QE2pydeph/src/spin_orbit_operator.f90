!
!   MODULE  :  spin orbit operator
!
MODULE spin_orbit_operator
  
  USE kinds,                        ONLY : DP
  USE pseudo_types,                 ONLY : pseudo_upf
  !
  
  TYPE (pseudo_upf), ALLOCATABLE, TARGET :: frpp (:)
  !
  !  fully relativistic pp MUST include SOC
  
  !
CONTAINS
  !
  ! =======================================================
  SUBROUTINE read_FR_pseudo (printout)
    ! -----------------------------------------------------
    
    USE radial_grids,                        ONLY : radial_grid_type, nullify_radial_grid
    USE pseudo_types,                        ONLY : nullify_pseudo_upf
    USE ions_base,                           ONLY : ntyp => nsp
    USE io_global,                           ONLY : ionode, stdout
    
    !
    implicit none
    !
    LOGICAL, OPTIONAL, INTENT(IN)                   :: printout
    !  information on atomic radial grid
    type(radial_grid_type), allocatable, target     :: rgrid (:)
    !
    integer, allocatable                            :: msh (:)
    integer                                         :: iunps
    integer                                         :: nt
    !
    LOGICAL                                         :: printout_ = .false.
    
    !
    !  ... allocate local radial grid
    !
    iunps = 4
    !
    ALLOCATE ( rgrid (ntyp), msh (ntyp) )

    !
    DO nt= 1, ntyp
       call nullify_radial_grid ( rgrid (nt) )
    END DO

    !
    ALLOCATE ( frpp (ntyp) )
    !
    DO nt= 1, ntyp
       call nullify_pseudo_upf ( frpp (nt) )
    END DO

    !
    IF ( PRESENT (printout) ) THEN
       printout_ = printout .AND. ionode
    END IF
    IF ( printout_ ) THEN
       WRITE( stdout,"(//,3X,'Atomic FR Pseudopotentials Parameters',/, &
            &    3X,'----------------------------------' )" )
    END IF
    WRITE(stdout,*) 'all set'
    
    
    !
  END SUBROUTINE read_FR_pseudo
  !
  ! =======================================================
  SUBROUTINE set_spin_orbit_operator ()
    ! -----------------------------------------------------

    USE constants,        ONLY : sqrt2
    USE uspp_param,       ONLY : upf, nh
    USE parameters,       ONLY : lmaxx
    USE ions_base,        ONLY : ntyp => nsp
    USE io_global,        ONLY : stdout
    USE uspp,             ONLY : indv, qq_at, qq_nt, qq_so
    USE spin_orb,         ONLY : fcoef, rot_ylm
    
    !
    implicit none
    
    !    internal variables
    
    integer                    :: nt, ih
    integer                    :: vi
    integer                    :: l, m
    integer                    :: n, n1
    
    
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
          !D_so = (0.d0,0.d0)
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
