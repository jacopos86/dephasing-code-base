!
!     PHYSICAL PARAMETERS MODULE
!
MODULE physical_constants
  !
  USE kinds,                 ONLY : DP
  
  
  
  !
  real(DP), parameter             :: gamma_e = 176085963023.0
  ! electron gyromagnetic ratio
  ! s^-1 T^-1
  real(DP), parameter             :: hbar = 1.0545718e-34
  ! h / 2 pi
  ! J s
  real(DP), parameter             :: ge = -2.00231930436256
  ! electron g factor
  real(DP), parameter             :: mu0 = 1.25663706212e-06
  ! magnetic moment constant
  ! N C^-2 s^2
  real(DP)                        :: D0
  ! prefactor
  ! MHz * bohr^3

  !
  ! units conversion factors
  !

  real(DP), parameter             :: m_to_bohr = 1.889725989e10
  ! m -> bohr
  ! conversion factor
  real(DP), parameter             :: Hz_to_joule = 6.62607015e-34
  ! Hz -> Joule
  
  
  !
CONTAINS
  !
  ! ======================================================================
  !
  !         Hss = S D S
  !
  !         Dab = 1/4 * mu_0/4pi * (gamma_e hbar)^2 \int rho(1,2) fab(1,2)
  !
  ! =======================================================================
  SUBROUTINE compute_prefactor ( )
    !
    USE constants,              ONLY : fpi
    
    !
    IMPLICIT NONE
    
    
    
    !
    D0 = 1._dp / 4
    D0 = D0 * mu0 / fpi
    D0 = D0 * (gamma_e * hbar) ** 2
    !
    ! units : J m^3
    D0 = D0 * m_to_bohr ** 3
    !
    ! units : MHz bohr^3
    D0 = D0 / Hz_to_joule * 1.e-6
    
    !
    RETURN
    !
  END SUBROUTINE compute_prefactor
  
  !
END MODULE physical_constants
