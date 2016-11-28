import unittest
import numpy as np

import utils
import core.linearSystem_htd_TotFixedDT_NEW as lsNEW

class TestInitialPropagateAndComputeBifRBCsIndex(unittest.TestCase):

    def test_bifRBCIndex(self):
        e={}
        e['v']=2.
        e['length']=30. #differene lengths

        e['rRBC']=np.linspace(0.,20.,101)
        print e['rRBC']

        #signs=np.array([-1,0,1])
        signs=np.array([-1,0,1])

        G = utils.createTestGraph()
        obj = lsNEW.LinearSystemHtdTotFixedDT_NEW(G, ht0=0.2) 
        obj._dt = 0.2
        for sign in signs:
            print sign
            indices = obj._initial_propagate_and_compute_bifRBCsIndex(e, sign)
            print len(indices)
            self.assertTrue(len(indices) == 0)

        #self.assertEqual('foo'.upper(), 'FOO')
        #self.assertFalse('Foo'.isupper())

if __name__ == '__main__':
    unittest.main()
