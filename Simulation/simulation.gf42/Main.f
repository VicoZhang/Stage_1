!=======================================================================
! Generated by  : PSCAD v4.6.2.0
!
! Warning:  The content of this file is automatically generated.
!           Do not modify, as any changes made here will be lost!
!-----------------------------------------------------------------------
! Component     : Main
! Description   : 
!-----------------------------------------------------------------------


!=======================================================================

      SUBROUTINE MainDyn()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 'emtstor.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's2.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'fnames.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'matlab.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------

!     SUBR    3PHVSRC       ! 3-Phase Source model
      INTEGER MRUNNUM       ! Gets Relative Run Number for Multiple Run Variables

!---------------------------------------
! Variable Declarations
!---------------------------------------


! Subroutine Arguments

! Electrical Node Indices

! Control Signals
      INTEGER  IT_1, FaultType, TransitionR
      REAL     Ua, Ub, Uc, FaultTime, ENAB, U(3)
      REAL     I(3), Ia, Ib, Ic, Is(3), Us(3)

! Internal Variables
      LOGICAL  LVD1_1
      INTEGER  IVD1_1
      REAL     RVD1_1, RVD1_2, RVD1_3, RVD1_4

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER ISTOI, ISTOF, IT_0                  ! Storage Indices
      INTEGER IPGB                                ! Control/Monitoring
      INTEGER ISUBS, SS(1), IBRCH(1), INODE       ! SS/Node/Branch/Xfmr
      INTEGER IXFMR


!---------------------------------------
! Local Indices
!---------------------------------------

! Dsdyn <-> Dsout transfer index storage

      NTXFR = NTXFR + 1

      TXFR(NTXFR,1) = NSTOL
      TXFR(NTXFR,2) = NSTOI
      TXFR(NTXFR,3) = NSTOF
      TXFR(NTXFR,4) = NSTOC

! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      ISTOI     = NSTOI
      NSTOI     = NSTOI + 3
      ISTOF     = NSTOF
      NSTOF     = NSTOF + 20
      IPGB      = NPGB
      NPGB      = NPGB + 9
      INODE     = NNODE + 2
      NNODE     = NNODE + 60
      IXFMR     = NXFMR
      NXFMR     = NXFMR + 3
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

! Initialize Subsystem Mapping

      ISUBS = NSUBS + 0
      NSUBS = NSUBS + 1

      DO IT_0 = 1,1
         SS(IT_0) = SUBS(ISUBS + IT_0)
      END DO

! Initialize Branch Mapping.

      IBRCH(1)     = NBRCH(SS(1))
      NBRCH(SS(1)) = NBRCH(SS(1)) + 105
!---------------------------------------
! Transfers from storage arrays
!---------------------------------------

      IT_1     = STOI(ISTOI + 1)
      Ua       = STOF(ISTOF + 1)
      Ub       = STOF(ISTOF + 2)
      Uc       = STOF(ISTOF + 3)
      FaultTime = STOF(ISTOF + 4)
      FaultType = STOI(ISTOI + 2)
      TransitionR = STOI(ISTOI + 3)
      ENAB     = STOF(ISTOF + 5)
      Ia       = STOF(ISTOF + 12)
      Ib       = STOF(ISTOF + 13)
      Ic       = STOF(ISTOF + 14)

! Array (1:3) quantities...
      DO IT_0 = 1,3
         U(IT_0) = STOF(ISTOF + 5 + IT_0)
         I(IT_0) = STOF(ISTOF + 8 + IT_0)
         Is(IT_0) = STOF(ISTOF + 14 + IT_0)
         Us(IT_0) = STOF(ISTOF + 17 + IT_0)
      END DO


!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


!---------------------------------------
! Configuration of Models
!---------------------------------------

      IF ( TIMEZERO ) THEN
         FILENAME = 'Main.dta'
         CALL EMTDC_OPENFILE
         SECTION = 'DATADSD:'
         CALL EMTDC_GOTOSECTION
      ENDIF
!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[source_3] Three Phase Voltage Source Model 2 'Source 1'
! Three Phase Source: Source 1  Type: L
!  
      RVD1_1 = RTCF(NRTCF)
      RVD1_2 = RTCF(NRTCF+1)
      RVD1_3  = 0.0*PI_BY180
      RVD1_4 = RTCF(NRTCF+3)
      NRTCF  = NRTCF + 4
      CALL EMTDC_3PHVSRC(SS(1), (IBRCH(1)+41), (IBRCH(1)+42), (IBRCH(1)+&
     &43), RVD1_4, .TRUE., RVD1_1 , RVD1_2, RVD1_3)
!

! 60:[time-sig] Output of Simulation Time 
      ENAB = TIME

! 100:[mrun] Multiple Run Component 
! ----------------------------------------
! Multiple Run Initialization
! ----------------------------------------
      CALL COMPONENT_ID(ICALL_NO,591962410)
      CALL MRUNVINI(72,1,410)


! List Variation of First Multiple Run Parameter:
! 4 Runs
      IF ( MRUNNUM(4,1)  .EQ. 1  ) FaultType = 1
      IF ( MRUNNUM(4,1)  .EQ. 2  ) FaultType = 4
      IF ( MRUNNUM(4,1)  .EQ. 3  ) FaultType = 7
      IF ( MRUNNUM(4,1)  .EQ. 4  ) FaultType = 8
!  Record Values into Output File"
      CALL MRUNVI(0,4,1,1,3,3,5,4,1,2,0,0,2.0,FaultType,"FaultType")



! Random or Sequential Variation of Second Multiple Run Parameter:
! 6 Runs
      CALL MRUNVR(0,4,1,2,3,3,5,6,4,0,0.05,0.02,2.0,FaultTime,"FaultTime&
     &")



! List Variation of Third Multiple Run Parameter:
! 3 Runs
      IF ( MRUNNUM(3,24)  .EQ. 1  ) TransitionR = 750
      IF ( MRUNNUM(3,24)  .EQ. 2  ) TransitionR = 1000
      IF ( MRUNNUM(3,24)  .EQ. 3  ) TransitionR = 1250
!  Record Values into Output File"
      CALL MRUNVI(0,4,1,3,3,3,5,3,24,2,0,0,2.0,TransitionR,"TransitionR"&
     &)






! 110:[pgb] Output Channel 'TransitionR'

      PGB(IPGB+7) = REAL(TransitionR)

! 120:[pgb] Output Channel 'FaultTime'

      PGB(IPGB+8) = FaultTime

! 130:[pgb] Output Channel 'FaultType'

      PGB(IPGB+9) = REAL(FaultType)

! 140:[tfaultn] Timed Fault Logic 
! Timed fault logic
      IT_1 = 0
      IF ( TIME .GE. FaultTime ) IT_1 = 1
      IF ( TIME .GE. (FaultTime+0.04) ) IT_1 = 0

! 150:[tpflt] Three Phase Fault 
      CALL E3PHFLT1_EXE(SS(1), (IBRCH(1)+50), (IBRCH(1)+51), (IBRCH(1)+5&
     &2), (IBRCH(1)+53), (IBRCH(1)+54), (IBRCH(1)+55),0,IT_1,FaultType,0&
     &.01)
      LVD1_1 = (OPENBR( (IBRCH(1)+50),SS(1)).AND.OPENBR( (IBRCH(1)+51),S&
     &S(1)).AND.OPENBR( (IBRCH(1)+52),SS(1)).AND.OPENBR( (IBRCH(1)+53),S&
     &S(1)).AND.OPENBR( (IBRCH(1)+54),SS(1)).AND.OPENBR( (IBRCH(1)+55),S&
     &S(1)))
      IVD1_1 = E_BtoI(LVD1_1)
      IF(FIRSTSTEP .OR. (IVD1_1 .NE. STORI(NSTORI))) THEN
         CALL PSCAD_AGI2(ICALL_NO,381336637,1-IVD1_1,"AG1")
         STORI(NSTORI) = IVD1_1
      ENDIF
      NSTORI = NSTORI + 1

! 160:[varrlc] Variable R, L or C  
      CALL E_VARRLC1_EXE(0 ,SS(1) ,  (IBRCH(1)+1), 0, REAL(TransitionR),&
     & 0.0)
      CALL E_VARRLC1_EXE(0 ,SS(1) ,  (IBRCH(1)+2), 0, REAL(TransitionR),&
     & 0.0)
      CALL E_VARRLC1_EXE(0 ,SS(1) ,  (IBRCH(1)+3), 0, REAL(TransitionR),&
     & 0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+31), (IBRCH(1)+32))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+29), (IBRCH(1)+30))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+27), (IBRCH(1)+28))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+25), (IBRCH(1)+26))

! 1:[xfmr-3p2w] 3 Phase 2 Winding Transformer 
!  TRANSFORMER SATURATION SUBROUTINE
      IVD1_1 = NEXC
      CALL TSAT2_EXE((IXFMR + 1),(IXFMR + 2),(IXFMR + 3), (IBRCH(1)+15),&
     & (IBRCH(1)+16), (IBRCH(1)+17), (IBRCH(1)+18), (IBRCH(1)+19), (IBRC&
     &H(1)+20),0,0,0,0,0,0,SS(1),0,1.0,0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+23), (IBRCH(1)+24))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+21), (IBRCH(1)+22))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+59), (IBRCH(1)+60))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+49), (IBRCH(1)+48))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+47), (IBRCH(1)+40))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+39), (IBRCH(1)+38))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+37), (IBRCH(1)+36))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+35), (IBRCH(1)+34))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+33), (IBRCH(1)+8))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+7), (IBRCH(1)+6))

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_EXE(SS(1), (IBRCH(1)+5), (IBRCH(1)+4))

!---------------------------------------
! Feedbacks and transfers to storage
!---------------------------------------

      STOI(ISTOI + 1) = IT_1
      STOF(ISTOF + 1) = Ua
      STOF(ISTOF + 2) = Ub
      STOF(ISTOF + 3) = Uc
      STOF(ISTOF + 4) = FaultTime
      STOI(ISTOI + 2) = FaultType
      STOI(ISTOI + 3) = TransitionR
      STOF(ISTOF + 5) = ENAB
      STOF(ISTOF + 12) = Ia
      STOF(ISTOF + 13) = Ib
      STOF(ISTOF + 14) = Ic

! Array (1:3) quantities...
      DO IT_0 = 1,3
         STOF(ISTOF + 5 + IT_0) = U(IT_0)
         STOF(ISTOF + 8 + IT_0) = I(IT_0)
         STOF(ISTOF + 14 + IT_0) = Is(IT_0)
         STOF(ISTOF + 17 + IT_0) = Us(IT_0)
      END DO


!---------------------------------------
! Transfer to Exports
!---------------------------------------

!---------------------------------------
! Close Model Data read
!---------------------------------------

      IF ( TIMEZERO ) CALL EMTDC_CLOSEFILE
      RETURN
      END

!=======================================================================

      SUBROUTINE MainOut()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 'emtstor.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's2.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'fnames.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'matlab.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------

      REAL    EMTDC_VVDC    ! 

!---------------------------------------
! Variable Declarations
!---------------------------------------


! Electrical Node Indices
      INTEGER  NT_27(3), NT_33(3)

! Control Signals
      REAL     Ua, Ub, Uc, ENAB, U(3), I(3), Ia
      REAL     Ib, Ic, Is(3), Us(3)

! Internal Variables
      INTEGER  IVD1_1

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER ISTOL, ISTOI, ISTOF, ISTOC, IT_0    ! Storage Indices
      INTEGER IPGB                                ! Control/Monitoring
      INTEGER ISUBS, SS(1), IBRCH(1), INODE       ! SS/Node/Branch/Xfmr
      INTEGER IXFMR


!---------------------------------------
! Local Indices
!---------------------------------------

! Dsdyn <-> Dsout transfer index storage

      NTXFR = NTXFR + 1

      ISTOL = TXFR(NTXFR,1)
      ISTOI = TXFR(NTXFR,2)
      ISTOF = TXFR(NTXFR,3)
      ISTOC = TXFR(NTXFR,4)

! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      IPGB      = NPGB
      NPGB      = NPGB + 9
      INODE     = NNODE + 2
      NNODE     = NNODE + 60
      IXFMR     = NXFMR
      NXFMR     = NXFMR + 3
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

! Initialize Subsystem Mapping

      ISUBS = NSUBS + 0
      NSUBS = NSUBS + 1

      DO IT_0 = 1,1
         SS(IT_0) = SUBS(ISUBS + IT_0)
      END DO

! Initialize Branch Mapping.

      IBRCH(1)     = NBRCH(SS(1))
      NBRCH(SS(1)) = NBRCH(SS(1)) + 105
!---------------------------------------
! Transfers from storage arrays
!---------------------------------------

      Ua       = STOF(ISTOF + 1)
      Ub       = STOF(ISTOF + 2)
      Uc       = STOF(ISTOF + 3)
      ENAB     = STOF(ISTOF + 5)
      Ia       = STOF(ISTOF + 12)
      Ib       = STOF(ISTOF + 13)
      Ic       = STOF(ISTOF + 14)

! Array (1:3) quantities...
      DO IT_0 = 1,3
         U(IT_0) = STOF(ISTOF + 5 + IT_0)
         I(IT_0) = STOF(ISTOF + 8 + IT_0)
         Is(IT_0) = STOF(ISTOF + 14 + IT_0)
         Us(IT_0) = STOF(ISTOF + 17 + IT_0)
      END DO


!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


! Array (1:3) quantities...
      DO IT_0 = 1,3
         NT_27(IT_0) = NODE(INODE + 36 + IT_0)
         NT_33(IT_0) = NODE(INODE + 43 + IT_0)
      END DO

!---------------------------------------
! Configuration of Models
!---------------------------------------

      IF ( TIMEZERO ) THEN
         FILENAME = 'Main.dta'
         CALL EMTDC_OPENFILE
         SECTION = 'DATADSO:'
         CALL EMTDC_GOTOSECTION
      ENDIF
!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[source_3] Three Phase Voltage Source Model 2 'Source 1'
      Ia = ( CBR((IBRCH(1)+41), SS(1)))
      Ib = ( CBR((IBRCH(1)+42), SS(1)))
      Ic = ( CBR((IBRCH(1)+43), SS(1)))

! 20:[multimeter] Multimeter 
      IVD1_1 = NRTCF
      NRTCF  = NRTCF + 5
      Is(1) = ( CBR((IBRCH(1)+44), SS(1)))
      Is(2) = ( CBR((IBRCH(1)+45), SS(1)))
      Is(3) = ( CBR((IBRCH(1)+46), SS(1)))
      Us(1) = EMTDC_VVDC(SS(1), NT_27(1), 0)
      Us(2) = EMTDC_VVDC(SS(1), NT_27(2), 0)
      Us(3) = EMTDC_VVDC(SS(1), NT_27(3), 0)

! 30:[multimeter] Multimeter 
      IVD1_1 = NRTCF
      NRTCF  = NRTCF + 5
      I(1) = ( CBR((IBRCH(1)+56), SS(1)))
      I(2) = ( CBR((IBRCH(1)+57), SS(1)))
      I(3) = ( CBR((IBRCH(1)+58), SS(1)))
      U(1) = EMTDC_VVDC(SS(1), NT_33(1), 0)
      U(2) = EMTDC_VVDC(SS(1), NT_33(2), 0)
      U(3) = EMTDC_VVDC(SS(1), NT_33(3), 0)

! 40:[pgb] Output Channel 'U'

      DO IVD1_1 = 1, 3
         PGB(IPGB+1+IVD1_1-1) = 1000.0 * U(IVD1_1)
      ENDDO

! 50:[pgb] Output Channel 'I'

      DO IVD1_1 = 1, 3
         PGB(IPGB+4+IVD1_1-1) = 1000.0 * I(IVD1_1)
      ENDDO

! 70:[datatap] Scalar/Array Tap 
      Ua = U(1)

! 80:[datatap] Scalar/Array Tap 
      Ub = U(2)

! 90:[datatap] Scalar/Array Tap 
      Uc = U(3)

! 100:[mrun] Multiple Run Component 

! --------------------------------------------------------
! Multiple Run Recording Initialization
! --------------------------------------------------------
      CALL COMPONENT_ID(ICALL_NO,591962410)
      CALL MRUNOINI(3,3,5,"mrunout.out")

! Multiple Run Recording

      CALL MRUNOR(0,4,1,1,3,3,5,0,NINT(ENAB),Ua,"U_A")

      CALL MRUNOR(0,4,1,2,3,3,5,0,NINT(ENAB),Ub,"U_B")

      CALL MRUNOR(0,4,1,3,3,3,5,0,NINT(ENAB),Uc,"U_C")




! 150:[tpflt] Three Phase Fault 
!
! Multi-phase Fault Currents
!
!

!---------------------------------------
! Feedbacks and transfers to storage
!---------------------------------------

      STOF(ISTOF + 1) = Ua
      STOF(ISTOF + 2) = Ub
      STOF(ISTOF + 3) = Uc
      STOF(ISTOF + 5) = ENAB
      STOF(ISTOF + 12) = Ia
      STOF(ISTOF + 13) = Ib
      STOF(ISTOF + 14) = Ic

! Array (1:3) quantities...
      DO IT_0 = 1,3
         STOF(ISTOF + 5 + IT_0) = U(IT_0)
         STOF(ISTOF + 8 + IT_0) = I(IT_0)
         STOF(ISTOF + 14 + IT_0) = Is(IT_0)
         STOF(ISTOF + 17 + IT_0) = Us(IT_0)
      END DO


!---------------------------------------
! Close Model Data read
!---------------------------------------

      IF ( TIMEZERO ) CALL EMTDC_CLOSEFILE
      RETURN
      END

!=======================================================================

      SUBROUTINE MainDyn_Begin()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------


!---------------------------------------
! Variable Declarations
!---------------------------------------


! Subroutine Arguments

! Electrical Node Indices

! Control Signals

! Internal Variables
      INTEGER  IVD1_1
      REAL     RVD1_1, RVD1_2, RVD1_3, RVD1_4
      REAL     RVD1_5, RVD1_6

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER IT_0                                ! Storage Indices
      INTEGER ISUBS, SS(1), IBRCH(1), INODE       ! SS/Node/Branch/Xfmr
      INTEGER IXFMR


!---------------------------------------
! Local Indices
!---------------------------------------


! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      INODE     = NNODE + 2
      NNODE     = NNODE + 60
      IXFMR     = NXFMR
      NXFMR     = NXFMR + 3
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

! Initialize Subsystem Mapping

      ISUBS = NSUBS + 0
      NSUBS = NSUBS + 1

      DO IT_0 = 1,1
         SS(IT_0) = SUBS(ISUBS + IT_0)
      END DO

! Initialize Branch Mapping.

      IBRCH(1)     = NBRCH(SS(1))
      NBRCH(SS(1)) = NBRCH(SS(1)) + 105
!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[source_3] Three Phase Voltage Source Model 2 'Source 1'
      IF (0.1 .LE. 1.0E-38) THEN
        CALL E_BRANCH_CFG( (IBRCH(1)+41),SS(1),1,0,0,1.0E-38,0.0,0.0)
        CALL E_BRANCH_CFG( (IBRCH(1)+42),SS(1),1,0,0,1.0E-38,0.0,0.0)
        CALL E_BRANCH_CFG( (IBRCH(1)+43),SS(1),1,0,0,1.0E-38,0.0,0.0)
      ELSE
        CALL E_BRANCH_CFG( (IBRCH(1)+41),SS(1),0,1,0,0.0,0.1,0.0)
        CALL E_BRANCH_CFG( (IBRCH(1)+42),SS(1),0,1,0,0.0,0.1,0.0)
        CALL E_BRANCH_CFG( (IBRCH(1)+43),SS(1),0,1,0,0.0,0.1,0.0)
      ENDIF
      RTCF(NRTCF)   = 110.0*SQRT_2*SQRT_1BY3
      RTCF(NRTCF+1) = 50.0*TWO_PI
      RTCF(NRTCF+3) = 0.0
      NRTCF = NRTCF + 4

! 60:[time-sig] Output of Simulation Time 

! 110:[pgb] Output Channel 'TransitionR'

! 120:[pgb] Output Channel 'FaultTime'

! 130:[pgb] Output Channel 'FaultType'

! 150:[tpflt] Three Phase Fault 
      CALL E3PHFLT1_CFG(1000000.0,0.0)

! 160:[varrlc] Variable R, L or C  
      CALL E_VARRLC1_CFG(0 ,SS(1) ,  (IBRCH(1)+1), 0)
      CALL E_VARRLC1_CFG(0 ,SS(1) ,  (IBRCH(1)+2), 0)
      CALL E_VARRLC1_CFG(0 ,SS(1) ,  (IBRCH(1)+3), 0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[xfmr-3p2w] 3 Phase 2 Winding Transformer 
      CALL COMPONENT_ID(ICALL_NO,122378945)
      RVD1_1 = ONE_3RD*60.0
      RVD1_2 = 110.0*SQRT_1BY3
      RVD1_3 = 11.0
      CALL E_TF2W_CFG((IXFMR + 1),0,RVD1_1,50.0,0.1,0.0,RVD1_2,RVD1_3,1.&
     &0)
      CALL E_TF2W_CFG((IXFMR + 2),0,RVD1_1,50.0,0.1,0.0,RVD1_2,RVD1_3,1.&
     &0)
      CALL E_TF2W_CFG((IXFMR + 3),0,RVD1_1,50.0,0.1,0.0,RVD1_2,RVD1_3,1.&
     &0)
      IF (0.0 .LT. 1.0E-6) THEN
        RVD1_5 = 0.0
        RVD1_6 = 0.0
        IVD1_1 = 0
      ELSE
        RVD1_6 = 0.0
        RVD1_4 = 6.0/(60.0*RVD1_6)
        RVD1_5 = RVD1_4*RVD1_2*RVD1_2
        RVD1_6 = RVD1_4*RVD1_3*RVD1_3
        IVD1_1 = 1
      ENDIF
      CALL E_BRANCH_CFG( (IBRCH(1)+9),SS(1),IVD1_1,0,0,RVD1_5,0.0,0.0)
      CALL E_BRANCH_CFG( (IBRCH(1)+10),SS(1),IVD1_1,0,0,RVD1_5,0.0,0.0)
      CALL E_BRANCH_CFG( (IBRCH(1)+11),SS(1),IVD1_1,0,0,RVD1_5,0.0,0.0)
      CALL E_BRANCH_CFG( (IBRCH(1)+12),SS(1),IVD1_1,0,0,RVD1_6,0.0,0.0)
      CALL E_BRANCH_CFG( (IBRCH(1)+13),SS(1),IVD1_1,0,0,RVD1_6,0.0,0.0)
      CALL E_BRANCH_CFG( (IBRCH(1)+14),SS(1),IVD1_1,0,0,RVD1_6,0.0,0.0)
      CALL TSAT2_CFG(2, (IBRCH(1)+15), (IBRCH(1)+16), (IBRCH(1)+17), (IB&
     &RCH(1)+18), (IBRCH(1)+19), (IBRCH(1)+20),0,0,0,0,0,0,SS(1),RVD1_1,&
     &0.2,1.17,50.0,0.0,1.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,RVD1_2,RVD1_3,0.&
     &0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

! 1:[fixed_load] Fixed Load 
      CALL LOAD1P1_CFG(11.0,50.0,20.0,20.0,2.0,2.0,0.0,0.0)

      RETURN
      END

!=======================================================================

      SUBROUTINE MainOut_Begin()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------


!---------------------------------------
! Variable Declarations
!---------------------------------------


! Subroutine Arguments

! Electrical Node Indices
      INTEGER  NT_27(3), NT_33(3)

! Control Signals

! Internal Variables
      INTEGER  IVD1_1

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER IT_0                                ! Storage Indices
      INTEGER ISUBS, SS(1), IBRCH(1), INODE       ! SS/Node/Branch/Xfmr
      INTEGER IXFMR


!---------------------------------------
! Local Indices
!---------------------------------------


! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      INODE     = NNODE + 2
      NNODE     = NNODE + 60
      IXFMR     = NXFMR
      NXFMR     = NXFMR + 3
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

! Initialize Subsystem Mapping

      ISUBS = NSUBS + 0
      NSUBS = NSUBS + 1

      DO IT_0 = 1,1
         SS(IT_0) = SUBS(ISUBS + IT_0)
      END DO

! Initialize Branch Mapping.

      IBRCH(1)     = NBRCH(SS(1))
      NBRCH(SS(1)) = NBRCH(SS(1)) + 105
!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


! Array (1:3) quantities...
      DO IT_0 = 1,3
         NT_27(IT_0) = NODE(INODE + 36 + IT_0)
         NT_33(IT_0) = NODE(INODE + 43 + IT_0)
      END DO

!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 20:[multimeter] Multimeter 
      IVD1_1 = NRTCF
      NRTCF  = NRTCF + 5

! 30:[multimeter] Multimeter 
      IVD1_1 = NRTCF
      NRTCF  = NRTCF + 5

! 40:[pgb] Output Channel 'U'

! 50:[pgb] Output Channel 'I'

! 70:[datatap] Scalar/Array Tap 

! 80:[datatap] Scalar/Array Tap 

! 90:[datatap] Scalar/Array Tap 

      RETURN
      END

