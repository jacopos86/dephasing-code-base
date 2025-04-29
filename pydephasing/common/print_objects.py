import yaml
from pydephasing.log import log
from pydephasing.set_param_object import p

#   This module prints 
#   different objects

#
# function -> print matrix
#

def print_2D_matrix(A):
	size = A.shape
	for i in range(size[0]):
		line = ""
		for j in range(size[1]):
			line += "  {0:.3f}".format(A[i,j])
		log.info("\t " + line)
	log.info("\n")
	log.info("\t " + p.sep)

# write spin vector on file

def write_time_dependent_data_on_file(time, data, data_label, file_name):
	# time steps
	nt = len(time)
	# set dictionary
	dict = {'nt' : 0, 'time' : 0, 'Mt' : 0}
	dict['time'] = time
	dict['nt'] = nt
	dict[data_label] = data
	# save data
	with open(file_name, 'w') as out_file:
		yaml.dump(dict, out_file)

#
# function -> print ZPL gradient data
# on output file
#

def print_zpl_fluct(gradZPL, hessZPL, out_dir):
	# write tensor to file
	file_name = "ZPL_grad.yml"
	file_name = "{}".format(out_dir + '/' + file_name)
	data = {'gradZPL' : gradZPL, 'hessZPL' : hessZPL}
	# eV / ang - eV / ang^2 units
	with open(file_name, 'w') as out_file:
		yaml.dump(data, out_file)