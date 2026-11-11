import os
import unittest
import tempfile
import numpy as np
import torch
from src.dataset import GlacierDataset

class TestDataset(unittest.TestCase):
    def test_glacier_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            patch = np.random.rand(8, 256, 256).astype(np.float32)
            mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
            
            np.save(os.path.join(td, "sample_000_patch.npy"), patch)
            np.save(os.path.join(td, "sample_000_mask.npy"), mask)
            
            dataset = GlacierDataset(td)
            self.assertEqual(len(dataset), 1)
            
            img, msk = dataset[0]
            self.assertEqual(img.shape, (8, 256, 256))
            self.assertEqual(msk.shape, (1, 256, 256))

if __name__ == "__main__":
    unittest.main()
