import unittest
import myflow as mf
import numpy as np


class TestOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = mf.Session()
        cls.a = np.asarray([[1, 4, 3], [7, 9, 0]])
        cls.b = np.asarray([[2, 1, 2], [3, 3, 2]])
        cls.A = mf.constant(cls.a)
        cls.B = mf.constant(cls.b)

    @classmethod
    def tearDownClass(cls):
        print("complete")

    def test_divide(self):
        c=TestOps.A/TestOps.B
        c_val=TestOps.sess.run([c])[0]
        print(c_val)

    def test_reduce_sum(self):
        c=mf.reduce_sum(TestOps.A, axis=0)
        c_val=TestOps.sess.run([c])[0]
        print(c_val)

    def test_dropout(self):
        c=mf.dropout(TestOps.B, keep_prob=0.8)
        c_val=TestOps.sess.run([c])[0]
        print(c_val)



