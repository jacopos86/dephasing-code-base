from distutils.core import setup
#
setup(name='pydephasing',
	packages=['pydephasing'],
	version='1.0.0',
	license='gpl-3.0',
	description='The PyDephasing code is designed to compute the inhomogeneous pure dephasing time of spin qubits and the excitonic dephasing time. It takes in input data from VASP and phonopy and it works as a post-processing tool for this data.',
	author='Jacopo Simoni',
	author_email='jsimoni@lbl.gov',
	url='',
	keywords=['inhomogeneous dephasing', 'zero field splitting', 'excitonic gap', 'hyperfine interaction', 'DFT'],
	install_requires=['numpy', 'pandas', 'matplotlib', 'pymatgen', 'mpi4py', 'statsmodels'],
)
