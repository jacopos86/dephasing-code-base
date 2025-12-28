#!/bin/bash

# Remove old config if exists
[ -e config2.yml ] && rm config.yml

wd=$(pwd)

# Read LOG_LEVEL from first argument

LOG_LEVEL=$1

# Exit if not provided
if [ -z "$LOG_LEVEL" ]; then
    echo "ERROR: LOG_LEVEL must be provided"
    exit 1
fi

echo "Using LOG_LEVEL: $LOG_LEVEL"

# colored or file logging

COLOR_LOG=$2
LOG_FILE=$3

if [ -z "$COLOR_LOG" ]; then
	echo "ERROR: COLOR_LOG must be provided"
	exit 1
fi
echo "COLORED_LOGGING : $COLOR_LOG"

if [ $COLOR_LOG -eq 0 ]; then
	if [ -z "$LOG_FILE" ]; then
		echo "ERROR: LOG_FILE must be provided"
		exit 1
	fi
fi

# log file

if [ $COLOR_LOG -eq 0 ]; then
	echo "WRITE TO LOGFILE : $LOG_FILE"
fi

# GPU

INSTALL_PYCUDA=$4
GPU_ACTIVE=$INSTALL_PYCUDA
GPU_BLOCK_SIZE=$5
GPU_GRID_SIZE=$6

if [ $GPU_ACTIVE -eq 0 ]; then
	echo "Run on CPU"
elif [ $GPU_ACTIVE -eq 1 ]; then
	echo "Run on GPU"
	# Check GPU_BLOCK_SIZE
	BLOCK_ARR=($GPU_BLOCK_SIZE)
	if [ ${#BLOCK_ARR[@]} -ne 3 ]; then
		echo "Error: GPU_BLOCK_SIZE must have exactly 3 elements"
		exit 1
	fi
	echo "GPU block size: ${BLOCK_ARR[@]}"
	# Check GPU_GRID_SIZE
	GRID_ARR=($GPU_GRID_SIZE)
	if [ ${#GRID_ARR[@]} -ne 2 ]; then
		echo "Error: GPU_GRID_SIZE must have exactly 2 elements"
		exit 1
	fi
	echo "GPU grid size: ${GRID_ARR[@]}"
fi

# update input test file paths
# if clean tests

BUILD_TESTS=$7
TESTS_12_TAR_FILE=$8
TESTS_3_TAR_FILE=$9

# Example: export BUILD_TESTS="1 2 3"

BUILD_ARR=($BUILD_TESTS)

# Check if 1 or 2 is present
if [[ " ${BUILD_ARR[@]} " =~ " 1 " ]] || [[ " ${BUILD_ARR[@]} " =~ " 2 " ]]; then
	    
	# -------------------------
	# Build TESTS if requested
	# -------------------------
	    
	echo "Build tests 1 / 2"
	
	if [ -d "$wd/EXAMPLES" ]; then
		echo "EXAMPLES directory already exists"
	elif [ -f $TESTS_12_TAR_FILE ]; then
		echo "Extracting EXAMPLES.tar.gz ..."
            	tar -xzf $TESTS_12_TAR_FILE
            	echo "Extraction complete."
    	else
        	echo "EXAMPLES.tar.gz missing"
        	exit 1
    	fi
    	
    	# TESTS directory
    	# TEST 1
    	
    	if [[ " ${BUILD_ARR[@]} " =~ " 1 " ]]; then
		if [ -d "$wd/TESTS/1" ]; then
			rm -rf "$wd/TESTS/1"
		fi
		mkdir -p "$wd/TESTS/1"
	
		cd "$wd/TESTS/1"
		wdT1=$(pwd)
		
		# write test 1 input		
		
cat > $wdT1/input.yml <<EOF
working_dir : ${wdT1}
unpert_dir : GS
displ_poscar_dir :
   - DISPLACEMENT-FILES-001
displ_outcar_dir :
   - DISPL-001
copy_files_dir : COPY-FOLDER
displ_ang :
   - - 0.01
     - 0.01
     - 0.01
max_dab : 2.7
defect_index : 0
max_dist_from_defect : 5.0
EOF
		
		DIR=${wdT1}/"GS"
		if [ ! -d "$DIR" ]; then
			cp -r ${wd}/EXAMPLES/C-CENTER/GS ${wdT1}
		fi
		
		DIR=${wdT1}/"COPY-FOLDER"
		if [ ! -d "$DIR" ]; then
			cp -r ${wd}/EXAMPLES/C-CENTER/COPY-FOLDER ${wdT1}
		fi
		
		cd ${wd}
	fi
	
	# TEST 2 -> pydephasing - NV center
	
	if [[ " ${BUILD_ARR[@]} " =~ " 2 " ]]; then
		if [ -d "$wd/TESTS/2" ]; then
			rm -rf "$wd/TESTS/2"
		fi
		mkdir -p "$wd/TESTS/2"
		
		cd "$wd/TESTS/2"
		wdT2=$(pwd)
		
		# write TEST 2 input files
		# input A
		
cat > $wdT2/inputA.yml <<EOF
working_dir : ${wdT2}
output_dir : ${wdT2}/T2-SP-DEPHC_A
displ_poscar_dir :
   - DISPLACEMENT-FILES-01
   - DISPLACEMENT-FILES-0001
displ_2nd_poscar_dir :
   - DISPLACEMENT-FILES-2NDORDER
displ_outcar_dir :
   - DISPL-01
   - DISPL-0001
displ_2nd_outcar_dir :
   - DISPL-2NDORDER
grad_info_file : info.yml
unpert_dir : GS
yaml_pos_file : phonopy_disp.yaml
hd5_eigen_file : mesh-nosymm_3x3x3.hdf5
2nd_order_correct : False
atom_res : False
phonon_res : False
nwg : 1000
eta : 6.6E-8
min_freq : 1.E-2
temperature :
   - 1.0
   - 10.0
   - 100.0
   - 200.0
   - 300.0
   - 400.0
B0 :
   - 0.0
   - 0.0
   - 1.0
EOF
		
		# input B
		
cat > $wdT2/inputB.yml <<EOF
working_dir : ${wdT2}
output_dir : ${wdT2}/T2-SP-DEPHC_B
displ_poscar_dir :
   - DISPLACEMENT-FILES-01
   - DISPLACEMENT-FILES-0001
displ_2nd_poscar_dir :
   - DISPLACEMENT-FILES-2NDORDER
displ_outcar_dir :
   - DISPL-01
   - DISPL-0001
displ_2nd_outcar_dir :
   - DISPL-2NDORDER
grad_info_file : info.yml
unpert_dir : GS
yaml_pos_file : phonopy_disp.yaml
hd5_eigen_file : mesh-nosymm_3x3x3.hdf5
2nd_order_correct : True
hessian : False
atom_res : False
phonon_res : False
T :
   - 1.0
   - 0.5
dt : 0.0007
T2_extract_method : fit
min_freq : 1.0E-2
eta : 6.6E-8
lorentz_thres : 1.0E-8
temperature :
   - 1.0
   - 10.0
   - 100.0
   - 200.0
   - 300.0
   - 400.0
B0 :
   - 0.0
   - 0.0
   - 1.0
EOF
		
		# input C
		
cat > $wdT2/inputC.yml <<EOF
working_dir : ${wdT2}
output_dir : ${wdT2}/T2-SP-DEPHC_C
displ_poscar_dir :
   - DISPLACEMENT-FILES-01
   - DISPLACEMENT-FILES-0001
displ_outcar_dir :
   - DISPL-01
   - DISPL-0001
grad_info_file : info.yml
unpert_dir : GS
yaml_pos_file : phonopy_disp.yaml
hd5_eigen_file : mesh-nosymm_3x3x3.hdf5
2nd_order_correct : True
atom_res : False
phonon_res : False
min_freq : 1.E-2
T :
   - 10.0
   - 1.0
dt : 0.0007
dynamics :
   - 0
   - 0
temperature :
   - 10.0
B0 :
   - 0.0
   - 0.0
   - 1.0
Bt :
   var : t
   expr_x : '0'
   expr_y : '0'
   expr_z : 1e-6*sin(1e-5*t+pi/2)
psi0 :
   - 1.0
   - 1.0
   - 0.0
EOF
		
		# input D
		
cat > $wdT2/inputD.yml <<EOF
working_dir : ${wdT2}
output_dir : ${wdT2}/T2-SP-DEPHC_D
displ_poscar_dir :
   - DISPLACEMENT-FILES-01
   - DISPLACEMENT-FILES-0001
displ_2nd_poscar_dir :
   - DISPLACEMENT-FILES-2NDORDER
displ_outcar_dir :
   - DISPL-01
   - DISPL-0001
displ_2nd_outcar_dir :
   - DISPL-2NDORDER
grad_info_file : info.yml
unpert_dir : GS
yaml_pos_file : phonopy_disp.yaml
hd5_eigen_file : mesh-nosymm_3x3x3.hdf5
2nd_order_correct : True
hessian : True
atom_res : False
phonon_res : False
nwg : 1000
w_max : 10.0
eta : 6.6E-8
min_freq : 1.E-2
temperature :
   - 1.0
   - 10.0
   - 100.0
   - 200.0
   - 300.0
   - 400.0
B0 :
   - 0.0
   - 0.0
   - 1.0
EOF
		
		# input E
		
cat > $wdT2/inputE.yml <<EOF
working_dir : ${wdT2}
output_dir : ${wdT2}/T2-SP-DEPHC_E
displ_poscar_dir :
   - DISPLACEMENT-FILES-01
   - DISPLACEMENT-FILES-0001
displ_2nd_poscar_dir :
   - DISPLACEMENT-FILES-2NDORDER
displ_outcar_dir :
   - DISPL-01
   - DISPL-0001
displ_2nd_outcar_dir :
   - DISPL-2NDORDER
grad_info_file : info.yml
unpert_dir : GS
yaml_pos_file : phonopy_disp.yaml
hd5_eigen_file : mesh-nosymm_3x3x3.hdf5
2nd_order_correct : True
hessian : False
atom_res : False
phonon_res : False
nwg : 1000
w_max : 10.0
eta : 6.6E-8
min_freq : 1.E-2
temperature :
   - 1.0
   - 10.0
   - 100.0
   - 200.0
   - 300.0
   - 400.0
B0 :
   - 0.0
   - 0.0
   - 1.0
EOF
		
		# copy all required files
		
		DIR=${wdT2}/"DISPLACEMENT-FILES-01"
		if [ ! -d "$DIR" ]; then
			cp -r ${wd}/EXAMPLES/NV-DIAMOND/DISPLACEMENT-FILES-01 ${wdT2}
			cp -r ${wd}/EXAMPLES/NV-DIAMOND/DISPLACEMENT-FILES-0001 ${wdT2}
			cp -r ${wd}/EXAMPLES/NV-DIAMOND/DISPLACEMENT-FILES-2NDORDER ${wdT2}
		fi
		
		DIR=${wdT2}/"DISPL-01"
		if [ ! -d "$DIR" ]; then
			cp -r ${wd}/EXAMPLES/NV-DIAMOND/DISPL-01 ${wdT2}
			cp -r ${wd}/EXAMPLES/NV-DIAMOND/DISPL-0001 ${wdT2}
 			cp -r ${wd}/EXAMPLES/NV-DIAMOND/DISPL-2NDORDER ${wdT2}
		fi
		
		DIR=${wdT2}/"GS"
		if [ ! -d "$DIR" ]; then
			cp -r ${wd}/EXAMPLES/NV-DIAMOND/GS ${wdT2}
		fi
		
		FIL=${wdT2}/"mesh-nosymm_3x3x3.hdf5"
		if [ ! -f "$FIL" ]; then
			cp ${wd}/EXAMPLES/NV-DIAMOND/info.yml ${wdT2}
			cp ${wd}/EXAMPLES/NV-DIAMOND/phonopy*.yaml ${wdT2}
			cp ${wd}/EXAMPLES/NV-DIAMOND/mesh-nosymm_3x3x3.hdf5 ${wdT2}
		fi

		echo -e "\nNN_model : MLP" >> ${wdT2}/info.yml
		echo -e "NN_parameters :" >> ${wdT2}/info.yml
		echo -e "  n_hidden_layers : !!python/tuple [100,]" >> ${wdT2}/info.yml
		echo -e "  solver : adam" >> ${wdT2}/info.yml
		echo -e "  activation : relu" >> ${wdT2}/info.yml
		echo -e "  alpha : 0.1" >> ${wdT2}/info.yml
		echo -e "  max_iter : 100" >> ${wdT2}/info.yml
		echo -e "  random_state : 1" >> ${wdT2}/info.yml
		echo -e "  test_size : 0.25" >> ${wdT2}/info.yml
		echo -e "  shuffle : True" >> ${wdT2}/info.yml
		
		# remove EXAMPLES directory
		
		rm -rf ${wd}/EXAMPLES
	
		cd ${wd}
		
	fi
fi
	
# TESTS 3

if [[ " ${BUILD_ARR[@]} " =~ " 3 " ]]; then
	if [ -d "$wd/TESTS/3" ]; then
		rm -rf "$wd/TESTS/3"
	fi
	
	echo "Build test 3"
	
	if [ -d "$wd/3" ]; then
		echo "TEST 3 directory already exists"
	elif [ -f $TESTS_3_TAR_FILE ]; then
		echo "Extracting TESTS_3.tar.gz ..."
            	tar -xzf $TESTS_3_TAR_FILE
            	echo "Extraction complete."
    	else
        	echo "TESTS_3.tar.gz missing"
        	exit 1
    	fi
    	
    	mv ${wd}/3 ${wd}/TESTS/
	
	wdT3=${wd}/TESTS/3	
	cd ${wdT3}
	
cat > $wdT3/input.yml <<EOF
working_dir : ${wdT3}
output_dir : ${wdT3}/T2-ELEC-DEPH
yaml_pos_file : phonopy_disp.yaml
unpert_dir : VASP_soc_relaxation
eph_matr_file : El-ph_data.npy
hd5_eigen_file : band.hdf5
2nd_order_correct : True
hessian : False
EOF

fi

echo "BUILD PROCEDURE COMPLETE"
