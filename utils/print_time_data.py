import yaml
import sys
import numpy as np
from pydephasing.log import log
import matplotlib.pyplot as plt
# 
# usage : python extract_data.py work_dir
#
work_dir =sys.argv[1]
# open file
try:
	f = open(work_dir+"/rho_oft.yml")
except:
	msg = "could not find: " + work_dir+"/rho_oft.yml"
	log.error(msg)
data = yaml.load(f, Loader=yaml.Loader)
f.close()
#
t = data['time']
rho_oft = data['quantity']
tr = np.zeros(rho_oft.shape[2])
for i in range(rho_oft.shape[0]):
	tr[:] += rho_oft[i,i,:].real
#
plt.plot(t, tr, '--')
plt.plot(t, rho_oft[0,0,:].real, label='[0,0]')
plt.plot(t, rho_oft[1,1,:].real, label='[1,1]')
plt.plot(t, rho_oft[2,2,:].real, label='[2,2]')
plt.legend()
plt.savefig('rho_diag.png')
plt.show()
#
plt.plot(t, rho_oft[0,1,:].real, label='[0,1]')
plt.plot(t, rho_oft[0,2,:].real, label='[0,2]')
plt.plot(t, rho_oft[1,2,:].real, label='[1,2]')
plt.legend()
plt.savefig('rho_outdiag.png')
plt.show()

# open file
try:
	f = open(work_dir+"/B_oft.yml")
except:
	msg = "could not find: " + work_dir+"/B_oft.yml"
	log.error(msg)
data = yaml.load(f, Loader=yaml.Loader)
f.close()
#
t = data['time']
B_oft = data['quantity']
#
plt.plot(t, B_oft[0,:].real, label=r'x')
plt.plot(t, B_oft[1,:].real, label=r'y')
plt.plot(t, B_oft[2,:].real, label=r'z')
plt.legend()
plt.savefig('B_oft.png')
plt.show()

# open file
try:
	f = open(work_dir+"/spin_vec.yml")
except:
	msg = "could not find: " + work_dir+"/spin_vec.yml"
	log.error(msg)
data = yaml.load(f, Loader=yaml.Loader)
f.close()
#
t = data['time']
S_oft = data['quantity']
#
plt.plot(t, S_oft[0,:].real, label=r'x')
plt.plot(t, S_oft[1,:].real, label=r'y')
plt.plot(t, S_oft[2,:].real, label=r'z')
plt.legend()
plt.savefig('S_oft.png')
plt.show()