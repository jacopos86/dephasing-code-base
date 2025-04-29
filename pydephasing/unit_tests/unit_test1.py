import os
import unittest
import pydephasing.unit_tests.test1 as test1

class TestProgram(unittest.TestCase):
    def setUp(self):
        self.TESTS_DIR = os.environ['TESTS']
        yml_inp = str(self.TESTS_DIR) + "/1/input.yml"
        self.argv_list = ["--yml_inp", yml_inp]
    def test_program_parse(self):
        # See "Things I've tried"
        args = test1.parse_args(self.argv_list)
        yml_inp = str(self.TESTS_DIR) + "/1/input.yml"
        self.assertEqual(yml_inp, args.yml_inp)
    def test_parameters(self):
        yml_inp = self.argv_list[1]
        pobj = test1.read_parameters(yml_inp)
        self.assertEqual(pobj.unpert_dir, str(self.TESTS_DIR)+"/1/GS")
        self.assertEqual(pobj.displ_poscar_dir[0], str(self.TESTS_DIR)+"/1/DISPLACEMENT-FILES-001")
        self.assertEqual(pobj.displ_outcar_dir[0], str(self.TESTS_DIR)+"/1/DISPL-001")
        self.assertEqual(pobj.atoms_displ[0][0], 0.01)
        self.assertEqual(pobj.atoms_displ[0][1], 0.01)
        self.assertEqual(pobj.atoms_displ[0][2], 0.01)
        self.assertEqual(pobj.defect_index, 0)

if __name__ == '__main__':
    unittest.main()