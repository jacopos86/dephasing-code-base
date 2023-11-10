!
!    SUBROUTINE to extract wave functions
!
SUBROUTINE extract_real_space_wfc ( ik, evc_r )
  
  !
  !    internal variables
  
  implicit none
  
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
END SUBROUTINE extract_real_space_wfc
