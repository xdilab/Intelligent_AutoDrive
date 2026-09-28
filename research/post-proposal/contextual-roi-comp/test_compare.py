import unittest
from compare import paired,METRICS
class Interaction(unittest.TestCase):
 def test_interaction_not_final_ranking(self):
  a,b,c,d=[dict.fromkeys(METRICS,n) for n in [10,12,15,16]]
  r=paired(a,b,c,d)
  self.assertEqual(r['new_arm_difference']['triplet'],4)
  self.assertEqual(r['interaction']['triplet'],-1)
 def test_positive_interaction_can_have_negative_gains(self):
  a,b,c,d=[dict.fromkeys(METRICS,n) for n in [10,8,11,10]]
  r=paired(a,b,c,d);self.assertEqual(r['interaction']['triplet'],1);self.assertEqual(r['contrastive_composition_gain']['triplet'],-1)
