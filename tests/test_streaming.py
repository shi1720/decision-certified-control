import unittest
import numpy as np
from benchmarks import history,query_state
from src.certified import Archive,DecisionCache,append_history
from src.portable import export_cache,evaluate

class StreamingChecks(unittest.TestCase):
 def test_remap_and_portability(self):
  full,_,_=history('nonnormal6',922,'uniform',192)
  old=Archive(full.center,full.radius,full.eta,full.Z[:64],full.Y[:64]);old.umax=full.umax
  cache=DecisionCache(old,1e-4);rng=np.random.default_rng(923)
  for t in range(8):cache.query(*query_state(old,rng,t))
  before=[(old.C[sorted(ids)].copy(),old.b[i,sorted(ids)].copy()) for i,ids in enumerate(cache.ids)]
  append_history(cache,full)
  for i,ids in enumerate(cache.ids):
   # IDs preserve the original actual inequalities even when the negative
   # data block moves. Compare row sets, independent of new ordering.
   actual={(tuple(full.C[j]),float(full.b[i,j])) for j in ids}
   expected={(tuple(c),float(b)) for c,b in zip(*before[i])}
   self.assertEqual(actual,expected)
  for t in range(8):
   x,v=query_state(full,rng,t);u,r=cache.query(x,v)
   payload=export_cache(cache)
   self.assertNotIn('Z',payload);self.assertNotIn('Y',payload)
   up,rp=evaluate(payload,x,v)
   np.testing.assert_allclose(u,up,atol=2e-6)
   self.assertLessEqual(rp['gap'],1e-4+1e-7)
 def test_nonextension_rejected(self):
  a,_,_=history('thermal4',924,'uniform',64);b,_,_=history('thermal4',925,'uniform',128)
  with self.assertRaises(ValueError):append_history(DecisionCache(a),b)
if __name__=='__main__':unittest.main()
