"""Contracts for model-independent SID components."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "SID modeling dependencies are not installed")
class SIDComponentTests(unittest.TestCase):
    def test_rqvae_forward_and_codebook_gradients(self):
        import torch
        from Representations.SID.rqvae import RQVAE

        model = RQVAE(4, hidden_dims=(8,), latent_dim=2, levels=3, codebook_size=4)
        loss, codes, _ = model(torch.randn(6, 4))
        self.assertEqual(tuple(codes.shape), (6, 3))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(codebook.grad is not None for codebook in model.codebooks))

    def test_collision_suffix_makes_ids_unique(self):
        from Representations.SID.scripts.generate_sids import unique_semantic_ids

        sids, groups, largest = unique_semantic_ids(
            ["B", "A", "C"], [[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        self.assertEqual(sids["A"], [1, 2, 3, 0])
        self.assertEqual(sids["B"], [1, 2, 3, 1])
        self.assertEqual(sids["C"], [4, 5, 6, 0])
        self.assertEqual((groups, largest), (1, 2))



if __name__ == "__main__":
    unittest.main()
