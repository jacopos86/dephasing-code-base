import yaml
import sys
from pydephasing.log import log
# 
# usage : python extract_data.py file_name dict_key
#
input_file =sys.argv[1]
dict_key   =sys.argv[2]
# open file
try:
	f = open(input_file)
except:
	msg = "could not find: " + input_file
	log.error(msg)
data = yaml.load(f, Loader=yaml.Loader)
f.close()
#
T = data['temperature']
nT = len(T)
if dict_key in data.keys():
	d = data[dict_key]
	nc = len(d)
	if d is None:
		msg = dict_key + " not computed"
		log.error(msg)
	if dict_key == 'Delt':
		for ic in range(nc):
			dd = d[ic]
			for iT in range(nT):
				log.info(" ic = " + str(ic+1) + " -  T = " + str(T[iT]) + " ( K ) ---------------- Delta = " + str(dd[iT]) + " eV")
	elif dict_key == 'T2':
		for iT in range(nT):
			for ic in range(nc):
				dd = d[ic]
				log.info(" ic = " + str(ic+1) + " -  T = " + str(T[iT]) + " ( K ) ---------------- T2 = " + str(dd[:,iT]) + " sec")
	elif dict_key == 'tau_c':
		for iT in range(nT):
			for ic in range(nc):
				dd = d[ic]
				log.info(" ic = " + str(ic+1) + " -  T = " + str(T[iT]) + " ( K ) ---------------- tau_c = " + str(dd[:,iT]) + " psec")
	elif dict_key == 'lw_eV':
		for iT in range(nT):
			for ic in range(nc):
				dd = d[ic]
				log.info(" ic = " + str(ic+1) + " -  T = " + str(T[iT]) + " ( K ) ---------------- lw = " + str(dd[:,iT]) + " eV")
else:
	msg = "wrong dictionary key"
	log.error(msg)