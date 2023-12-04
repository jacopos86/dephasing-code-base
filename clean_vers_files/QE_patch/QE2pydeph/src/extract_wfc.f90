!
!    SUBROUTINE to extract wave functions
!
SUBROUTINE extract_evc_r_gamma_only ( ik, ibnd, evc_r )
  
  USE kinds,                 ONLY : DP
  USE fft_base,              ONLY : dffts
  USE io_files,              ONLY : iunwfc, nwordwfc
  USE wavefunctions,         ONLY : evc, psic
  USE fft_interfaces,        ONLY : invfft
  USE fft_base,              ONLY : dffts
  USE klist,                 ONLY : nks, ngk
  USE wvfct,                 ONLY : nbnd
  USE control_flags,         ONLY : lxdm
  USE funct,                 ONLY : dft_is_meta
  
  implicit none
  
  !
  complex(DP), intent(out)        :: evc_r (dffts%nnr)
  integer, intent(in)             :: ik
  integer, intent(in)             :: ibnd
  
  !
  !    internal variables
  !
  
  integer                         :: ir
  integer                         :: npw
  LOGICAL                         :: use_tg
  
  !!  here extract k pt wave function
  !
  
  use_tg = ( dffts%has_task_groups ) .and. ( .not. (dft_is_meta() .or. lxdm) )
  !
  
  IF ( use_tg ) THEN
     !
     call errore ('extract_evc_r', 'task groups not implemented', 1)
     !
  END IF
  
  !
  !  plane waves
  
  npw = ngk (ik)
  !
  IF ( nks > 1 ) call get_buffer ( evc, nwordwfc, iunwfc, ik )
  
  !
  psic (:) = (0.d0,0.d0)
  !
  IF ( ibnd < nbnd ) THEN
     !
     !   FFT transform
     !
     psic (dffts%nl(1:npw)) = evc (1:npw,ibnd)
     psic (dffts%nlm(1:npw)) = CONJG ( evc (1:npw,ibnd) )
     !
  ELSE
     !
     call errore ('extract_evc_r', 'wrong band index', 1)
     !
  END IF
  
  !
  !   inv FFT wfc -> real space
  call invfft ( 'Wave', psic, dffts )
  
  !
  !   store data in evc_r
  
  do ir= 1, dffts%nnr
     evc_r (ir) = psic (ir)
  end do
  
  !
END SUBROUTINE extract_evc_r_gamma_only

! -------------------------------------------------------------
SUBROUTINE extract_evc_r_ofk ( ik, ibnd, evc_r )
  ! ===========================================================

  USE kinds,                ONLY : DP
  USE fft_base,             ONLY : dffts
  USE klist,                ONLY : igk_k, ngk, nks, xk
  USE wavefunctions,        ONLY : evc, psic
  USE control_flags,        ONLY : lxdm
  USE funct,                ONLY : dft_is_meta
  USE gvecw,                ONLY : ecutwfc
  USE gvect,                ONLY : g, ngm
  USE cell_base,            ONLY : tpiba2
  USE io_files,             ONLY : nwordwfc, iunwfc
  USE wvfct,                ONLY : npwx
  
  !
  implicit none

  integer, intent(in)           :: ik
  integer, intent(in)           :: ibnd
  complex(DP), intent(out)      :: evc_r (dffts%nnr)
  
  !
  !  internal variables
  
  integer                       :: ir, j
  integer                       :: npw
  LOGICAL                       :: use_tg
  real(DP), ALLOCATABLE         :: gk (:)
  real(DP)                      :: gcutw
  
  
  !!  here extract k pt wave function
  !
  
  use_tg = ( dffts%has_task_groups ) .and. ( .not. (dft_is_meta() .or. lxdm) )
  !
  
  IF ( use_tg ) THEN
     !
     call errore ('extract_evc_r', 'task groups not implemented', 1)
     !
  END IF
  
  !
  !   n. plane waves
  
  npw = ngk (ik)
  !
  IF (.NOT.ALLOCATED(igk_k) .AND. .NOT.ALLOCATED(ngk)) THEN
     !
     ALLOCATE( igk_k(npwx,nks) )
     ALLOCATE( ngk(nks) )
     ALLOCATE ( gk (npwx) )
     igk_k (:,:) = 0
     gcutw = ecutwfc/tpiba2
     
     !   call gk_sort
     
     call gk_sort ( xk (1,ik), ngm, g, gcutw, npw, igk_k (1,ik), gk )
     !
  END IF
  
  !
  IF ( nks > 1 ) call get_buffer ( evc, nwordwfc, iunwfc, ik )
  
  !
  call threaded_barrier_memset ( psic, 0.d0, dffts%nnr*2 )
  !
  do j= 1, npw
     psic (dffts%nl(igk_k(j,ik))) = evc (j,ibnd)
  end do
  
  !
  call invfft ('Wave', psic, dffts)
  
  !
  !   store data in evc_r
  
  do ir= 1, dffts%nnr
     evc_r (ir) = psic (ir)
  end do
  !
  
  RETURN
  !
END SUBROUTINE extract_evc_r_ofk
