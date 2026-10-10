import unittest
from sequence_metrics import batch_lengths,error_rates
class SequenceTests(unittest.TestCase):
 def test_short_and_final_batch(self):
  self.assertEqual(batch_lengths(1),[1]);self.assertEqual(batch_lengths(75),[75]);self.assertEqual(batch_lengths(76),[75,1])
 def test_no_frames(self):
  with self.assertRaises(ValueError):batch_lengths(0)
 def test_known_error_rates(self):
  self.assertEqual(error_rates('hello world','hello word')['wer'],.5)
  self.assertEqual(error_rates('abc','adc')['cer'],1/3)
