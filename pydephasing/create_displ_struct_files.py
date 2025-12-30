# this subroutine creates the input files
# to be run in the VASP calculation
from shutil import copyfile
from pydephasing.set_structs import UnpertStruct, DisplacedStructs, DisplacedStructures2ndOrder
import os
import yaml
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
#
def gen_poscars(max_dist_from_defect, defect_index):
    log.info("\n")
    log.info("\t " + p.sep)
    log.info("\t CREATE UNPERTURBED STRUCTURE")
    struct0 = UnpertStruct(p.unpert_dir)
    # read poscar
    struct0.read_poscar()
    # define all the displced atoms structures
    for i in range(len(p.displ_poscar_dir)):
        displ_struct = DisplacedStructs(p.displ_poscar_dir[i])
        # set atoms displacements
        log.debug("\t p.atoms_displ: " + str(p.atoms_displ))
        displ_struct.atom_displ(p.atoms_displ[i])   # Ang
        # build displaced atomic structures
        displ_struct.build_atom_displ_structs(struct0, max_dist_from_defect, defect_index)
        # write data on file
        displ_struct.write_structs_on_file()
        # delete structure object
        del displ_struct
    log.info("\t " + p.sep)
    log.info("\t POSCAR FILES WRITTEN")
    # create calculation directory
    nat = struct0.nat
    for i in range(len(p.displ_outcar_dir)):
        if not os.path.exists(p.displ_outcar_dir[i]):
            os.mkdir(p.displ_outcar_dir[i])
        # run over all calculations
        for ia in range(nat):
            # distance between atom and defect
            dd = struct0.struct.get_distance(ia, defect_index)
            if dd <= max_dist_from_defect:
                for idx in range(3):      # x-y-z index
                    for s in range(2):    # +/- index
                        namef = p.displ_poscar_dir[i] + "/POSCAR-" + str(ia+1) + "-" + str(idx+1) + "-" + str(s+1)
                        named = p.displ_outcar_dir[i] + "/" + str(ia+1) + "-" + str(idx+1) + "-" + str(s+1)
                        if not os.path.exists(named):
                            os.mkdir(named)
                        # copy files in new directory
                        files = os.listdir(p.copy_files_dir)
                        for fil in files:
                            copyfile(p.copy_files_dir + "/" + fil, named + "/" + fil)
                        copyfile(namef, named + "/POSCAR")
                        log.info("\t " + str(ia+1) + " - " + str(idx+1) + " - " + str(s+1) + " -> COMPLETED")
#
# generate 2nd order displ
# POSCAR
#
def gen_2ndorder_poscar(max_dist_from_defect, defect_index, max_d_ab):
    log.info("\n")
    log.info("\t " + p.sep)
    log.info("\t CREATE UNPERTURBED STRUCTURE")
    struct0 = UnpertStruct(p.unpert_dir)
    # read poscar
    struct0.read_poscar()
    # define displ. atomic structures
    for i in range(len(p.displ_poscar_dir)):
        # init structures
        displ_struct = DisplacedStructures2ndOrder(p.displ_poscar_dir[i])
        displ_struct.set_max_atoms_distance(max_d_ab)
        # set atoms displacement
        displ_struct.atom_displ(p.atoms_displ[i])   # ang
        # build displ. atoms structures
        displ_struct.build_atom_displ_structs(struct0, max_dist_from_defect, defect_index)
        # write data on file
        displ_struct.write_structs_on_file()
        log.info("\t " + p.sep)
        log.info("\t POSCAR FILES WRITTEN")
        # check if directory exists
        if not os.path.exists(p.displ_outcar_dir[i]):
            os.mkdir(p.displ_outcar_dir[i])
        # read summary files
        input_file = p.displ_poscar_dir[i] + "/summary.yml"
        try:
            f = open(input_file)
        except:
            msg = "\t COULD NOT FIND : " + input_file
            log.error(msg)
        data = yaml.load(f, Loader=yaml.Loader)
        dirlist = data['calc_list']
        # run over all calculations
        log.info("\n")
        log.info("\t " + p.sep)
        for named in dirlist:
            # check named exists
            out_dir = p.displ_outcar_dir[i]
            if not os.path.exists(out_dir + "/" + named):
                os.mkdir(out_dir + "/" + named)
            # copy files in new directory
            files = os.listdir(p.copy_files_dir)
            for fil in files:
                copyfile(p.copy_files_dir + "/" + fil, out_dir + "/" + named + "/" + fil)
            namef = p.displ_poscar_dir[i] + "/POSCAR-" + named
            copyfile(namef, out_dir + "/" + named + "/POSCAR")
            log.info("\t " + named + " -> COMPLETED")
        log.info("\t " + p.sep)