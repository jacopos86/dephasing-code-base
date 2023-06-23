import unittest
import numpy as np
from pydephasing.input_parameters import data_input
from pydephasing.compute_zfs_dephas import compute_homo_dephas

# set up testing unit

class testpydeph(unittest.TestCase):

    def test1(self):
        input_params = data_input()
        input_params.read_yml_data("./1/input_1.yml")
        T2_obj, Delt_obj, tauc_obj = compute_homo_dephas(input_params, False, False)
        # T2
        T2_res = np.array([0.23871798, 0.06163808])
        self.assertAlmostEqual(T2_obj.get_T2_sec()[0,0], T2_res[0])
        self.assertAlmostEqual(T2_obj.get_T2_sec()[1,0], T2_res[1])
        # Delta
        Delt_res = 1.90840864e-08
        self.assertAlmostEqual(Delt_obj.get_Delt()[0], Delt_res)
        # tau_c
        tauc_res = np.array([0.00498315, 0.01929922])
        self.assertAlmostEqual(tauc_obj.get_tauc()[0,0], tauc_res[0])
        self.assertAlmostEqual(tauc_obj.get_tauc()[1,0], tauc_res[1])
        
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()