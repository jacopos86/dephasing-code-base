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
        ! -----------------------------------------------
        SUBROUTINE init_nc_frpp ()
            ! -------------------------------------------


            !
            IMPLICIT NONE

            !
            ! run over atom types
            DO nt= 1, ntyp

                !
                OPEN ( UNIT = iunps, FILE = TRIM(file_pseudo), STATUS = 'old', FORM = 'formatted' )
                !
                ! read the norm-conserving PP

                IF ( pp_format (psfile (nt) ) == 5 ) THEN
                    !
                    IF( printout_ ) &
                        WRITE( stdout, "(3X,'file type is old PWscf NC format')")
                    call read_ncpp (iunps, nt, upf(nt))
                    !
                ELSE
                    !
                    call errore('readpp', 'file '//TRIM(file_pseudo)//' not readable',1)
                    !
                ENDIF
                !

                CLOSE (iunps)
                !
            END DO
            !

            RETURN
            !
        END SUBROUTINE init_nc_frpp