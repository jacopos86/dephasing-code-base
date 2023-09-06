# Makefile for postprocessing (ZFS_HFI)

include ../../make.inc

# location of needed modules and included files (if any)
MODFLAGS= $(BASEMOD_FLAGS) \
          $(MOD_FLAG)../../PW/src \
	  $(MOD_FLAG)../../PP/src \
	  $(MOD_FLAG)../../Modules \
          $(MOD_FLAG)../../dft-d3

ZFSOBJS = \
zfs_module.o \
constants_module.o

PWOBJS = ../../PW/src/libpw.a
PPOBJS = ../../PP/src/libpp.a
QEOBJS = ../../Modules/libqemod.a ../../KS_Solvers/libks_solvers.a \
         ../../FFTXlib/libqefft.a ../../LAXlib/libqela.a ../../UtilXlib/libutil.a \
         ../../dft-d3/libdftd3qe.a

MODULES = $(PWOBJS) $(QEOBJS) $(PPOBJS)

TLDEPS= pwlibs

all : tldeps zfs_calc_driver.x


libzfs.a : $(ZFSOBJS) 
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

zfs_calc_driver.x : zfs_calc_driver.o libzfs.a $(MODULES) $(LIBOBJS)
	$(LD) $(LDFLAGS) -o $@ \
		zfs_calc_driver.o libzfs.a $(MODULES) $(LIBOBJS) $(QELIBS)
	- ( cd ../../bin ; ln -fs ../PP/src/$@ . )

tldeps :
	if test -n "$(TLDEPS)" ; then \
	( cd ../.. ; $(MAKE) $(TLDEPS) || exit 1 ) ; fi

clean :
	- /bin/rm -f *.x *.o *~ *_tmp.f90 *.d *.mod *.i *.L libpp.a

include make.depend
