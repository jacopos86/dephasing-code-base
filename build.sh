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
unpert_dir :
   - GS
displ_poscar_dir :
   - DISPLACEMENT-FILES-001
displ_outcar_dir :
   - DISPL-001
copy_files_dir : COPY-FOLDER
displ_ang :
   - 0.01
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
exit

# test 2 -> pydephasing - NV center

cd ../
mkdir ./2
cd ./2
wdT2=$(pwd)
 cp ../../clean_vers_files/TESTS/2/inputA.yml ${wd}
 cp ../../clean_vers_files/TESTS/2/inputB.yml ${wd}
 DIR=${wd}/"DISPLACEMENT-FILES-01"
 if [ ! -d "$DIR" ]; then
 	cp -r ../../examples/NV-diamond/DISPLACEMENT-FILES-01 ${wd}
 	cp -r ../../examples/NV-diamond/DISPLACEMENT-FILES-0001 ${wd}
 	cp -r ../../examples/NV-diamond/DISPLACEMENT-FILES-2NDORDER ${wd}
 fi
 DIR=${wd}/"DISPL-01"
 if [ ! -d "$DIR" ]; then
 	cp -r ../../examples/NV-diamond/DISPL-01 ${wd}
 	cp -r ../../examples/NV-diamond/DISPL-0001 ${wd}
 	cp -r ../../examples/NV-diamond/DISPL-2NDORDER ${wd}
 fi
 DIR=${wd}/"GS"
 if [ ! -d "$DIR" ]; then
 	cp -r ../../examples/NV-diamond/GS ${wd}
 fi
 FIL=${wd}/"mesh-nosymm_3x3x3.hdf5"
 if [ ! -f "$FIL" ]; then
 	cp -r ../../examples/NV-diamond/info.yml ${wd}
 	cp -r ../../examples/NV-diamond/phonopy*.yaml ${wd}
 	cp -r ../../examples/NV-diamond/mesh-nosymm_3x3x3.hdf5 ${wd}
 fi
 sed -i "s,LOCAL-PATH,$wd," ./inputA.yml
 sed -i "s,LOCAL-PATH,$wd," ./inputB.yml
