import yaml
import sys
from utilities.log import log
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
	if d is None:
		msg = dict_key + " not computed"
		log.error(msg)
	if dict_key == 'Delt':
		for iT in range(nT):
			log.info(" T = " + str(T[iT]) + " ( K ) ---------------- Delta = " + str(d[iT]) + " eV")
	elif dict_key == 'T2':
		for iT in range(nT):
			log.info(" T = " + str(T[iT]) + " ( K ) ---------------- T2 = " + str(d[:,iT]) + " sec")
	elif dict_key == 'tau_c':
		for iT in range(nT):
			log.info(" T = " + str(T[iT]) + " ( K ) ---------------- tau_c = " + str(d[:,iT]) + " psec")
	elif dict_key == 'lw_eV':
		for iT in range(nT):
			log.info(" T = " + str(T[iT]) + " ( K ) ---------------- lw = " + str(d[:,iT]) + " eV")
else:
	msg = "wrong dictionary key"
	log.error(msg)