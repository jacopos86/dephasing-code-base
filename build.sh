#!/bin/bash

[ -e config2.yml ] && rm config.yml

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
 
# fix gpu source files headers path
 
wd=$(pwd)
wd+='/pydephasing/gpu_source'
sed -i "/extern_func.cuh/c\#include \"$wd/extern_func.cuh\"" $wd/compute_acf_V1.cu
sed -i "/extern_func.cuh/c\#include \"$wd/extern_func.cuh\"" $wd/compute_acf_V2.cu
exit

# update input test file paths
 
 # test 1 -> init
 cd ./TESTS/1
 wd=$(pwd)
 cp ../../clean_vers_files/TESTS/1/input.yml ${wd}
 DIR=${wd}/"GS"
 if [ ! -d "$DIR" ]; then
 	cp -r ../../examples/CC-TEST/GS ${wd}
 fi
 DIR=${wd}/"COPY-FOLDER"
 if [ ! -d "$DIR" ]; then
 	cp -r ../../examples/CC-TEST/COPY-FOLDER ${wd}
 fi
 sed -i "s,LOCAL-PATH,$wd," ./input.yml
 
 # test 2 -> pydephasing - NV center
 cd ../2
 wd=$(pwd)
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
 
 # QE2pydeph
 
 cd ../../
 cp ./clean_vers_files/Makefile .
 echo "install QE2pydeph? "
 read QE2pydeph
 if [ "$QE2pydeph" = "yes" ] || [ "$QE2pydeph" = "y" ]
 then
 	echo "local QE path :"
 	read QEpath
 	sed -i "s,LOCAL-PATH,$QEpath," ./Makefile
 	DIR=${QEpath}/"QE2pydeph"
 	if [ -d "$DIR" ]; then
  		echo "copy examples? "
 		read cp_examples
 		if [ "$cp_examples" = "yes" ] || [ "$cp_examples" = "y" ]
 		then
 			cp -r ./clean_vers_files/QE_patch/QE2pydeph/examples ${DIR}
 		fi
 		cp -r ./clean_vers_files/QE_patch/QE2pydeph/Doc ${DIR}
 		cp ./clean_vers_files/QE_patch/QE2pydeph/src/*.f90 ${DIR}/src/
 		cp ./clean_vers_files/QE_patch/QE2pydeph/Makefile ${DIR}
 		cp ./clean_vers_files/QE_patch/QE2pydeph/README.md ${DIR}
 	else
 		cp -r ./clean_vers_files/QE_patch/QE2pydeph/QE2pydeph ${QEpath}
 	fi
 elif [ "$QE2pydeph" = "no" ] || [ "$QE2pydeph" = "n" ]
 then
 	QEpath=""
 	sed -i "s,LOCAL-PATH,$QEpath," ./Makefile
 else
 	echo "answer: y or n"
 	sleep 3
 	exit 1
 fi
