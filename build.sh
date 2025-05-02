#!/bin/bash

[ -e config2.yml ] && rm config.yml

wd=$(pwd)

# log level
echo "log level: "
read log_level
echo "LOG_LEVEL : $log_level" >> config.yml 
 
# colored or file logging
echo "redirect to log file? (y/n)"
read set_log_file
if [ "$set_log_file" = "yes" ] || [ "$set_log_file" = "y" ]
then
	color_logging="FALSE"
elif [ "$set_log_file" = "no" ] || [ "$set_log_file" = "n" ]
then
	color_logging="TRUE"
else
	echo "answer: y or n"
	sleep 3
	exit 1
fi
echo "COLORED_LOGGING : $color_logging" >> config.yml
 
# log file
if [ "$color_logging" = "FALSE" ]
then
	echo "log file name: "
	read log_file
	echo "LOGFILE : $log_file" >> config.yml
fi
 
# GPU
echo "perform GPU calculation? "
read gpu_calc
if [ "$gpu_calc" = "yes" ] || [ "$gpu_calc" = "y" ]
then
	GPU="TRUE"
elif [ "$gpu_calc" = "no" ] || [ "$gpu_calc" = "n" ]
then
	GPU="FALSE"
else
	echo "answer: y or n"
	sleep 3
	exit 1
fi
echo "GPU : $GPU" >> config.yml
if [ "$GPU" = "TRUE" ]
then
	echo "GPU_BLOCK_SIZE :" >> config.yml
	echo "GPU block size (nbx): "
	read nbx
	echo "  - $nbx" >> config.yml
	read nby
	echo "  - $nby" >> config.yml
	read nbz
	echo "  - $nbz" >> config.yml
	#
	echo "GPU_GRID_SIZE :" >> config.yml
	echo "GPU grid size (ngx): "
	read ngx
	echo "  - $ngx" >> config.yml
	read ngy
	echo "  - $ngy" >> config.yml
fi

# update input test file paths
# test 1 -> init

echo "clean TESTS directory? "
read clean_tests
if [ "$clean_tests" = "yes" ] || [ "$clean_tests" = "y" ]
then
	CLEAN_TESTS="TRUE"
elif [ "$clean_tests" = "no" ] || [ "$clean_tests" = "n" ]
then
	CLEAN_TESTS="FALSE"
else
	echo "answer: y or n"
	sleep 3
	exit 1
fi

if [ "$CLEAN_TESTS" = "TRUE" ]
then
	if [ ! -d ${wd}/EXAMPLES ]
	then
		tar -xvzf EXAMPLES.tar.gz
	fi
	if [ -d TESTS ]
	then
		rm -r TESTS
	fi
	mkdir TESTS
	mkdir TESTS/1
	cd ./TESTS/1
	wdT1=$(pwd)

	cat > input.yml <<EOF
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

	# test 2 -> pydephasing - NV center

	cd ../
	mkdir ./2
	cd ./2
	wdT2=$(pwd)

	cat > inputA.yml <<EOF
working_dir : ${wdT2}
output_dir : T2-SP-DEPHC_A
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

	cat > inputB.yml <<EOF
working_dir : ${wdT2}
output_dir : T2-SP-DEPHC_B
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
EOF

	cat > inputC.yml <<EOF
working_dir : ${wdT2}
output_dir : T2-SP-DEPHC_C
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
EOF

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
		cp -r ${wd}/EXAMPLES/NV-DIAMOND/info.yml ${wdT2}
		cp -r ${wd}/EXAMPLES/NV-DIAMOND/phonopy*.yaml ${wdT2}
		cp -r ${wd}/EXAMPLES/NV-DIAMOND/mesh-nosymm_3x3x3.hdf5 ${wdT2}
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

	rm -rf ${wd}/EXAMPLES
fi