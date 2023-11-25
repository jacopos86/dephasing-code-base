!
!     MODULE   :  bec initialization
!
MODULE bec_module
  !
  USE becmod,                       ONLY : bec_type
  
  !
  TYPE (bec_type), allocatable, target   :: bec_sp (:)
  !
  !   <beta|psi>
  
  
  
  
  
  !
  !
CONTAINS
  !
  ! ===========================================================
  subroutine allocate_bec_arrays ()
    ! ---------------------------------------------------------
    
    USE wvfct,                 ONLY : nbnd
    USE uspp,                  ONLY : nkb
    USE becmod,                ONLY : allocate_bec_type
    USE klist,                 ONLY : nks
    
    !
    implicit none
    
    
    !
    !     internal variables
    
    INTEGER                         :: ik
    
    !
    !     allocate bec_sp
    
    ALLOCATE ( bec_sp (1:nks) )
    
    !
    WRITE(6,*) "nkb= ", nkb
    WRITE(6,*) "nks= ", nks
    
    !
    do ik= 1, nks
       !
       call allocate_bec_type (nkb, nbnd, bec_sp (ik))
       !   
    end do
    !
    
    !
    RETURN
    !
  end subroutine allocate_bec_arrays

  !
  ! ==============================================
  SUBROUTINE compute_bec_array ( )
    ! --------------------------------------------
    
    USE lsda_mod,          ONLY : current_spin, lsda, isk
    USE klist,             ONLY : nks
    
    !
    IMPLICIT NONE
    
    !   internal variables
    
    INTEGER                    :: ik
    
    
    
    !
    k_loop: DO ik= 1, nks
       !
       IF (lsda) current_spin = isk(ik)
       
       !
       !  compute bec_sp (ik)
       !
       
    END DO k_loop
    
    
    
    
    
    
    
    
  END SUBROUTINE compute_bec_array
  
  !
END MODULE bec_module
