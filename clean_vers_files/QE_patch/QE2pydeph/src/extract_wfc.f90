!
!    SUBROUTINE to extract wave functions
!
SUBROUTINE extract_real_space_wfc ( ik, evc_r )
  
  USE kinds,                 ONLY : DP
  USE fft_base,              ONLY : dffts
  USE cell_base,             ONLY : tpiba2
  USE io_files,              ONLY : iunwfc, nwordwfc
  USE wavefunctions,         ONLY : evc
  USE klist,                 ONLY : nks, ngk, xk, igk_k
  USE wvfct,                 ONLY : nbnd, npwx
  USE gvect,                 ONLY : g, ngm
  USE gvecw,                 ONLY : ecutwfc
  USE fft_interfaces,        ONLY : invfft
  
  implicit none
  
  !
  !    internal variables
  integer, intent(in)             :: ik
  complex(DP), intent(inout)      :: evc_r (dffts%nnr,nks,nbnd)
  !
  integer                         :: ig, ib
  integer                         :: npw
  real(DP), allocatable           :: gk (:)
       ! |G+k|^2
  
  !
  !  plane waves
  !
  
  ALLOCATE ( gk (1:npwx) )
  
  !
  npw = ngk (ik)
  gk (:) = 0._DP
  
  !
  !  ecutwfc/tpiba2 = gcutw
  !
  
  call gk_sort ( xk (1,ik), ngm, g, ecutwfc/tpiba2, npw, igk_k (1,ik), gk )
  
  !
  !  read the wavefunction
  !
  
  call davcio (evc, 2*nwordwfc, iunwfc, ik, -1)
  
  !
  evc_r (:,ik,:) = cmplx (0._dp, 0._dp, kind=dp)
  !
  do ig= 1, npw
     evc_r (dffts%nl (igk_k(ig,ik)), ik, :) = evc (ig,:)
  end do
  
  !
  do ib= 1, nbnd
     call invfft ('Wave', evc_r (:,ik,ib), dffts)
  end do
  
  !
END SUBROUTINE extract_real_space_wfc
