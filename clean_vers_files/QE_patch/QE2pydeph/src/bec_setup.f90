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
    
    USE mp_bands,              ONLY : intra_bgrp_comm
    USE wvfct,                 ONLY : nbnd
    USE uspp,                  ONLY : nkb
    USE becmod,                ONLY : allocate_bec_type
    USE klist,                 ONLY : nks
    USE uspp,                  ONLY : dvan_so
    
    !
    implicit none
    
    
    !
    !     internal variables
    
    INTEGER                         :: ik
    
    !
    !     allocate bec_sp
    
    ALLOCATE ( bec_sp (1:nks) )
    WRITE(6,*) shape (bec_sp), nkb, shape(dvan_so)
    call stop_pp
    !
    WRITE(6,*) "nkb= ", nkb, nks
    do ik= 1, nks
       !
       WRITE(6,*) ik, nkb, nbnd
       call allocate_bec_type (nkb, nbnd, bec_sp (ik))
       !
       
    end do
    !
    
    !
    RETURN
    !
  end subroutine allocate_bec_arrays
  
  !
END MODULE bec_module
