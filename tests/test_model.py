import unittest
import torch
from models.attention_unet import AttentionUNet, CBAM, AttentionGate

class TestAttentionUNet(unittest.TestCase):
    def test_attention_unet_forward(self):
        model = AttentionUNet(in_channels=8, out_channels=1)
        model.eval()
        x = torch.randn(2, 8, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 1, 256, 256))
        self.assertFalse(torch.isnan(out).any())

    def test_cbam_module_forward(self):
        cbam = CBAM(in_planes=64)
        cbam.eval()
        x = torch.randn(2, 64, 32, 32)
        with torch.no_grad():
            out = cbam(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_attention_gate_with_cbam(self):
        ag = AttentionGate(F_g=128, F_l=128, F_int=64)
        ag.eval()
        g = torch.randn(2, 128, 32, 32)
        x = torch.randn(2, 128, 32, 32)
        with torch.no_grad():
            out = ag(g, x)
        self.assertEqual(out.shape, (2, 128, 32, 32))
        self.assertFalse(torch.isnan(out).any())

if __name__ == "__main__":
    unittest.main()
