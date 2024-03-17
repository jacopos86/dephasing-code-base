!
! ---------------------------------------------------------
!
!   spin orbit coupling -> norm conserving pp
!
! ---------------------------------------------------------
!
MODULE spin_orbit_operator_ncpp
    !
    USE kinds,               ONLY : DP




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
    ! ====================================================================
    SUBROUTINE compute_beta_ofG (npw_, igk_, q_, betal_q_)
        ! ----------------------------------------------------------------
        !
        !   beta_l(G) = (4pi/omega)^0.5 \int_0^infty dr r^2 fl(r) jl(Gr)
        !
        ! ----------------------------------------------------------------
        
        USE uspp_param,          ONLY : lmaxkb
        USE us,                  ONLY : spline_ps
        
        !
        IMPLICIT NONE

        !
        INTEGER, INTENT(IN)          :: npw_
        !  n. PWs

        
        
        !
        !    internal variables

        REAL(DP), ALLOCATABLE        :: qg (:), vq (:)
        REAL(DP), ALLOCATABLE        :: xdata (:)
        !
        INTEGER                      :: iq


        !
        IF (lmaxkb < 0) RETURN

        !
        !  set cache blocking size
        numblock = (npw_+blocksize-1)/blocksize
        !
        IF (spline_ps) THEN
            ALLOCATE ( xdata (nqx) )
            DO iq= 1, nqx
               xdata (iq) = (iq - 1) * dq
            END DO
        END IF

        !
        ALLOCATE ( qg (blocksize) )
        ALLOCATE ( vq (blocksize) )
        ALLOCATE ( gk (3,blocksize) )

        !
        ! run over iblock







        




        !
        RETURN
        !
    END SUBROUTINE compute_beta_ofG
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