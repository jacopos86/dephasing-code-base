# dephasing-code-base
# code functionalities
1- computation of T2 spin dephasing time in solid state qubits\
2- computation of inhomogeneous dephasing time T2*\
3- computation of homogeneous linewidth
# installation procedure
1- create your own environment\
conda create -n pydeph python=3.8 numpy matplotlib scipy\
conda activate pydeph\
2- conda install --channel conda-forge pymatgen
# if GPU device present
3- pip install pycuda\
4- pip install h5py\
5- pip install pyyaml\
6- pip install mpi4py\
7- pip install colorlog\
8- pip install scikit-learn\
9- pip install tqdm\
make install
