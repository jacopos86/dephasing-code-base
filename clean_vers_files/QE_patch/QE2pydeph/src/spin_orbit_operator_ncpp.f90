!
! ---------------------------------------------------------
!
!   spin orbit coupling -> norm conserving pp
!
! ---------------------------------------------------------
!
MODULE spin_orbit_operator_ncpp









    !
    CONTAINS
    !
    ! =========================================================================
    SUBROUTINE compute_Vso_nnp ( )
        ! ---------------------------------------------------------------------
        !
        !  fSO(G,G') = \sum_pm (pm ihbar) \sum_l 2/(2l+1) |beta_pm^l(G)>
        !         (delta E_pm^l)^-1 <beta_pm^l(G')| Pl'(G,G') GXG'/|G|/|G'|
        !
        !  VSO(n,n') = \sum_G,G' <n,G|fSO(G,G')|n',G'>

        IMPLICIT NONE
        
        !
        !   internal variables




















        !
        RETURN
        !
    END SUBROUTINE compute_Vso_nnp

    !
    ! ==============================================
    SUBROUTINE compute_Pl_derivative ()
        ! ------------------------------------------


        !
        IMPLICIT NONE

        !
        !   internal variables





        

        
        !
        RETURN
        !
    END SUBROUTINE compute_Pl_derivative
    !
END MODULE spin_orbit_operator_ncpp