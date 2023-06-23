from mpi4py import MPI
import numpy as np
import random
from pydephasing.global_params import MPI_ROOT
# MPI class
class MPI_obj:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = MPI_ROOT
    # collect array
    def collect_array(self, array):
        array_full = np.zeros(array.shape)
        array_list = self.comm.gather(array, root=self.root)
        if self.rank == self.root:
            for a in array_list:
                array_full = array_full + a
        array_full = self.comm.bcast(array_full, root=self.root)
        return array_full
    # collect list
    def collect_list(self, lst):
        list_full = []
        list_full = self.comm.gather(lst, root=self.root)
        lst = []
        if self.rank == self.root:
            for l1 in list_full:
                for l in l1:
                    lst.append(l)
        lst = self.comm.bcast(lst, root=self.root)
        return lst
    # collect atom displ array
    def collect_time_array(self, f_oft):
        nt = f_oft.shape[0]
        f_oft_full = np.zeros(nt, dtype=type(f_oft))
        f_oft_list = self.comm.gather(f_oft[:], root=self.root)
        if self.rank == self.root:
            for f_oft in f_oft_list:
                f_oft_full[:] += f_oft[:]
        f_oft = self.comm.bcast(f_oft_full, root=self.root)
        return f_oft
    # split array between processes
    def split_ph_modes(self, nq, nl):
        data = []
        for iq in range(nq):
            for il in range(nl):
                data.append((iq,il))
        data = np.array(data)
        loc_proc_list = np.array_split(data, self.size)
        return loc_proc_list[self.rank]
    # split list of data
    def split_list(self, list_data):
        loc_proc_list = np.array_split(np.array(list_data), self.size)
        return list(loc_proc_list[self.rank])
    # random split list
    def random_split(self, list_data):
        data = list(np.array(list_data))
        # divide data in approx. equal parts
        lengths = np.zeros(self.size, dtype=int)
        lengths[:] = len(data) / self.size
        rest = len(data) % self.size
        i = 0
        while rest > 0:
            lengths[i] += 1
            rest -= 1
            i += 1
        assert sum(lengths) == len(data)
        # partition the data
        loc_proc_list = None
        chunks = None
        if self.rank == self.root:
            chunks = []
            for i in range(self.size):
                lst = []
                lst = random.sample(data, lengths[i])
                for x in lst:
                    data.remove(x)
                chunks.append(lst)
            assert len(data) == 0
        self.comm.Barrier()
        loc_proc_list = self.comm.scatter(chunks, root=self.root)
        self.comm.Barrier()
        return loc_proc_list
    # finalize procedure
    def finalize_procedure(self):
        MPI.Finalize()
# mpi -> obj
mpi = MPI_obj()