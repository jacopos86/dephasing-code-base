!
!    MODULE : spin operators declaration
!
MODULE spin_operators
    !
    USE kinds,           ONLY : DP

    !
    !  spin orbit coupling operator

    complex(DP), allocatable :: HSO_a (:,:)

    !
    !  spin operators

    complex(DP)              :: sigma_x (2,2)
    complex(DP)              :: sigma_y (2,2)
    complex(DP)              :: sigma_z (2,2)



    !
    CONTAINS
    !
    ! ====================================================================================
    subroutine set_spin_operators ()
        ! ----------------------------------------------------------------------------------

        IMPLICIT NONE

        !
        !    sigma operators

        sigma_x (1,2) = cmplx (1._dp,0._dp)
        sigma_x (2,1) = cmplx (1._dp,0._dp)

        !
        sigma_y (1,2) = cmplx (0._dp, -1._dp)
        sigma_y (2,1) = cmplx (0._dp,  1._dp)

        !
        sigma_z (1,1) = cmplx (1._dp, 0._dp)
        sigma_z (2,2) = cmplx (-1._dp, 0._dp)

        !
    end subroutine set_spin_operators
    !
    !
    ! =============================================================================================
    SUBROUTINE compute_soc_matrix_elements ( transitions_list, ntr )
        ! -------------------------------------------------------------------------------------------
        !
        !    this subroutine computes the SOC matrix
        !    elements to be used into the ZFS
        !    calculation
        !
        !    < Psi_o^s | H_SO^a | Psi_n^s' > =
        !             \sum_uv < Psi_o | beta_u > [chi(s) sigma_a chi(s')] <beta_v|Psi_n> Dso_a(u,v)
        !

        USE bec_module,           ONLY : bec_sp
        USE ions_base,            ONLY : ntyp => nsp, nat, ityp
        USE uspp_param,           ONLY : nh
        USE control_flags,        ONLY : gamma_only
        USE io_global,            ONLY : stdout

        !
        IMPLICIT NONE

        integer, intent(in)            :: ntr
        integer, intent(in)            :: transitions_list (ntr,6)

        !    internal variables

        integer                        :: a, ih, jh, ikb, jkb, ijkb0, itr, na, nt
        integer                        :: ki, kpi, ni, oi, si, spi, isp
        complex(DP)                    :: s_xyz (3)
        !    spin vector
        INTEGER                        :: ierr

        !
        !    allocate H_SO^a_on

        allocate ( HSO_a (ntr, 4), stat=ierr )
        if (ierr/=0) call errore ('compute_soc_matrix_elements', 'allocating HSO_a', abs (ierr))
        HSO_a = (0.d0, 0.d0)

        !
        !    iterate over transitions
        !

        do itr= 1, ntr
            !
            ki = transitions_list (itr, 1)
            oi = transitions_list (itr, 2)
            si = transitions_list (itr, 3) 
            kpi= transitions_list (itr, 4)
            ni = transitions_list (itr, 5)
            spi= transitions_list (itr, 6) 
            isp= (si-1)*2 + spi

            !
            !
            !  iterate atom species
            ijkb0 = 0
            do nt= 1, ntyp
                do na= 1, nat
                    IF ( ityp (na) .eq. nt ) THEN
                        !
                        do ih= 1, nh(nt)
                            ikb = ijkb0 + ih
                            do jh= 1, nh(nt)
                                jkb = ijkb0 + jh
                                !
                                IF (gamma_only) THEN
                                    HSO_a (itr,isp) = HSO_a (itr,isp) +     &
                                        bec_sp (ki)%r (ikb,oi) * Vso (ih,jh,isp,nt) * bec_sp (kpi)%r (jkb,ni)
                                !
                                !WRITE(stdout,*) HSO_a (itr,a), bec_sp (ki)%r (ikb,oi), s_xyz (a) 
                                ELSE
                                    HSO_a (itr,isp) = HSO_a (itr,isp) +     &
                                        conjg (bec_sp (ki)%k (ikb,oi)) * Vso (ih,jh,isp,nt) * bec_sp (kpi)%k (jkb,ni)
                                !
                                END IF
                            end do
                        end do
                        ijkb0 = ijkb0 + nh (nt)
                        !
                    END IF
                end do
            end do
            !
        end do

        !
        RETURN
        !
    END SUBROUTINE compute_soc_matrix_elements
    
    !
END MODULE spin_operators