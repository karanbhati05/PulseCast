import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main

class Checks(unittest.TestCase):
    def test_future_does_not_change_earlier_forecast(self):
        y=np.arange(100,dtype=float)+np.sin(np.arange(100)); p=main.forecast(y,80); y[90:]+=10000; q=main.forecast(y,80); np.testing.assert_allclose(p[:10],q[:10])
    def test_recommendations_exclude_seen(self):
        events=np.array([[1,0,0,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,1]],dtype=float); self.assertNotIn(0,main.recommend(events,0,k=3))
    def test_knapsack_optimum(self):
        chosen=main.schedule([5,9,10],[2,3,4],5); self.assertEqual(sum([5,9,10][i] for i in chosen),14)

if __name__=="__main__": unittest.main()
