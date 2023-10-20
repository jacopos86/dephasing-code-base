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
 
 # update input test file paths
