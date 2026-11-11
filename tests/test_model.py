import unittest
import torch
from models.attention_unet import AttentionUNet

class TestAttentionUNet(unittest.TestCase):
    def test_attention_unet_forward(self):
        model = AttentionUNet(in_channels=8, out_channels=1)
        model.eval()
        x = torch.randn(2, 8, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 1, 256, 256))
        self.assertFalse(torch.isnan(out).any())

    def test_cbam_attention_unet_forward(self):
        model = AttentionUNet(in_channels=8, out_channels=1, use_cbam=True)
        model.eval()
        x = torch.randn(2, 8, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 1, 256, 256))
        self.assertFalse(torch.isnan(out).any())

if __name__ == "__main__":
    unittest.main()
