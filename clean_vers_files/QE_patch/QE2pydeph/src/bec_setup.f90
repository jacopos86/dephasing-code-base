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
  ! ==================================================================================
  SUBROUTINE compute_bec_array ( )
    ! --------------------------------------------------------------------------------
    
    USE lsda_mod,          ONLY : current_spin, lsda, isk
    USE klist,             ONLY : nks, ngk, igk_k, xk
    USE wavefunctions,     ONLY : evc
    USE uspp,              ONLY : nkb, vkb
    USE realus,            ONLY : real_space, invfft_orbital_gamma, calbec_rs_gamma,    &
         invfft_orbital_k, calbec_rs_k
    USE becmod,            ONLY : calbec
    USE io_files,          ONLY : iunwfc, nwordwfc
    USE io_global,         ONLY : stdout
    USE control_flags,     ONLY : gamma_only
    USE buffers,           ONLY : get_buffer
    USE wvfct,             ONLY : nbnd
    USE mp_bands,          ONLY : inter_bgrp_comm
    USE mp,                ONLY : mp_sum
    USE spin_orbit_operator, ONLY : Dso
    USE uspp_param,  ONLY : nhm
    
    !
    IMPLICIT NONE
    
    !   internal variables
    
    INTEGER                    :: ik, ibnd, ibnd_end, ibnd_start, this_bgrp_nbnd
    integer                    :: npw

    !   prepare bands division
    !
    CALL divide( inter_bgrp_comm, nbnd, ibnd_start, ibnd_end )
    this_bgrp_nbnd = ibnd_end - ibnd_start + 1
    !
    k_loop: DO ik= 1, nks
       !
       IF (lsda) current_spin = isk(ik)
       
       !
       !  compute bec_sp (ik)
       !
       
       IF (nks>1) call get_buffer (evc, nwordwfc, iunwfc, ik)
       
       !
       IF (nkb>0) call init_us_2 ( ngk(ik), igk_k(1,ik), xk(1,ik), vkb )

       npw = ngk(ik)
       IF ( .NOT. real_space ) THEN
          ! calbec computes becp = <vkb_i|psi_j>
          call calbec (npw, vkb, evc, bec_sp (ik))
       ELSE
          if (gamma_only) then
             do ibnd= ibnd_start, ibnd_end, 2
                call invfft_orbital_gamma (evc, ibnd, ibnd_end)
                call calbec_rs_gamma (ibnd, ibnd_end, bec_sp (ik)%r)
             end do
             call mp_sum (bec_sp (ik)%r, inter_bgrp_comm)
          else
             bec_sp (ik)%k = (0.d0,0.d0)
             do ibnd= ibnd_start, ibnd_end
                call invfft_orbital_k (evc, ibnd, ibnd_end)
                call calbec_rs_k (ibnd, ibnd_end)
             end do
             call mp_sum (bec_sp(ik)%k, inter_bgrp_comm)
          end if
       END IF
       !
    END DO k_loop
    
    WRITE(stdout,*) "    becp_sp   -> calculation completed"
    !
  END SUBROUTINE compute_bec_array
  
  !
END MODULE bec_module
