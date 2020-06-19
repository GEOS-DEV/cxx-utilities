import unittest

import numpy as np

import lvarrayPythonInterface


class PythonArrayTests(unittest.TestCase):

    def test_sorted_array_init(self):
        for lowerbound, upperbound in ((0,10), (1,5), (-15, 15), (100, 200)):
            array_from_c = lvarrayPythonInterface.set_sorted_array(lowerbound, upperbound)
            array_from_python = np.array(range(lowerbound, upperbound), dtype=array_from_c.dtype)
            self.assertTrue((array_from_c==array_from_python).all())

    def test_sorted_array_modification(self):
        for lowerbound, upperbound in ((0,10), (1,5), (-15, 15), (100, 200)):
            for factor in range(2, 6):
                array_from_c = lvarrayPythonInterface.set_sorted_array(lowerbound, upperbound)
                array_from_c *= factor
                array_from_python = np.array(range(factor * lowerbound, factor * upperbound, factor), dtype=array_from_c.dtype)
                self.assertTrue((array_from_c==array_from_python).all())
                self.assertTrue((lvarrayPythonInterface.get_sorted_array()==array_from_python).all())
                self.assertTrue((lvarrayPythonInterface.get_sorted_array()==array_from_c).all())

    def test_create_np_array(self):
        for lowerbound, upperbound in ((0,10), (1,5), (-15, 15), (100, 200)):
            array_from_c = lvarrayPythonInterface.create_np_array(lowerbound, upperbound)
            self.assertTrue((array_from_c==np.array(range(lowerbound, upperbound), dtype=array_from_c.dtype)).all())

    def test_create_np_array_bad_input(self):
        self.assertEqual(lvarrayPythonInterface.create_np_array(10, -1), None)
        with self.assertRaises(TypeError):
            lvarrayPythonInterface.create_np_array("hello", 2)
            lvarrayPythonInterface.create_np_array(1, 6.77)

if __name__ == '__main__':
    unittest.main()
