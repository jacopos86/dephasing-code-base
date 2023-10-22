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
 cp ./clean_vers_files/gpu_source/compute_acf_V1.cu ${wd}
 cp ./clean_vers_files/gpu_source/compute_acf_V2.cu ${wd}
 sed -i "s,LOCAL-PATH,$wd," ./pydephasing/gpu_source/compute_acf_V1.cu
 sed -i "s,LOCAL-PATH,$wd," ./pydephasing/gpu_source/compute_acf_V2.cu
 
 # update input test file paths
 
 cd ./TESTS/1
 wd=$(pwd)
 cp ../../clean_vers_files/TESTS/1/input.yml ${wd}
 sed -i "s,LOCAL-PATH,$wd," ./input.yml
 cd ../2
 wd=$(pwd)
 cp ../../clean_vers_files/TESTS/2/inputA.yml ${wd}
 cp ../../clean_vers_files/TESTS/2/inputB.yml ${wd}
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
