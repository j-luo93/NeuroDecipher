import numpy as np
import unittest

class TestEditDistance(unittest.TestCase):
    def test_editdistance(self):
        import editdistance
        self.assertEqual(2, editdistance.eval('abc', 'aec'))
        self.assertEqual(np.asarray([[2, 3], [1, 2]], dtype='int64').tolist(), editdistance.eval_all(['ab', 'abc'], ['bc', 'bcd']).tolist())

    def test_time(self):
        import uuid, editdistance
        strings = [uuid.uuid4().hex.lower()[0:6] for _ in range(5000)]
        editdistance.eval_all(strings, strings)

if __name__ == '__main__':
    unittest.main()
