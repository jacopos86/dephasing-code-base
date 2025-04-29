
#   This module prints 
#   different objects

# write spin vector on file
def write_spin_vector_on_file(self, out_dir):
		# time steps
		nt = len(self.time)
		# write on file
		namef = out_dir + "/spin-vector.yml"
		# set dictionary
		dict = {'time' : 0, 'Mt' : 0}
		dict['time'] = self.time
		dict['Mt'] = self.Mt
		# save data
		with open(namef, 'w') as out_file:
			yaml.dump(dict, out_file)
		# mag. vector
		namef = out_dir + "/occup-prob.yml"
		# set dictionary
		dict2 = {'time' : 0, 'occup' : 0}
		occup = np.zeros((3,nt))
		# run over t
		for i in range(nt):
			occup[:,i] = np.dot(self.tripl_psit[:,i].conjugate(), self.tripl_psit[:,i]).real
		dict2['time'] = self.time
		dict2['occup']= occup
		# save data
		with open(namef, 'w') as out_file:
			yaml.dump(dict2, out_file)