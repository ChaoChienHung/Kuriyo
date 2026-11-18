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

    def test_reset_dead_codes_reseeds_only_unused_codewords(self):
        import torch
        from Representations.SID.rqvae import RQVAE, nearest_code

        torch.manual_seed(0)
        model = RQVAE(4, hidden_dims=(8,), latent_dim=2, levels=2, codebook_size=4)
        inputs = torch.randn(16, 4)
        with torch.no_grad():
            for codebook in model.codebooks:
                codebook.zero_()
                codebook[1:] = 100.0  # far away: no input selects codes 1-3
        resets = model.reset_dead_codes(inputs, torch.Generator().manual_seed(0))
        self.assertEqual(resets[0], 3)
        latent = model.encoder(inputs)
        first = model.codebooks[0].detach()
        self.assertTrue(torch.equal(first[0], torch.zeros(2)))
        for row in first[1:]:
            self.assertTrue(torch.isclose(latent, row).all(-1).any())
        self.assertEqual(len({tuple(row.tolist()) for row in first[1:]}), 3)
        used = torch.bincount(nearest_code(latent, first), minlength=4)
        self.assertTrue(bool((used[1:] > 0).all()))
        self.assertEqual(model.reset_dead_codes(inputs)[0], 0)

    def test_adagrad_accumulator_controls_first_step_size(self):
        import torch
        from Representations.SID.rqvae import RQVAE
        from Representations.SID.scripts.generate_sids import build_optimizer

        inputs = torch.nn.functional.normalize(torch.randn(64, 32) + 3.0, dim=-1)
        norms = {}
        for accumulator in (0.0, 0.1):
            torch.manual_seed(0)
            model = RQVAE(32, hidden_dims=(64, 32), latent_dim=8, codebook_size=8)
            optimizer = build_optimizer("adagrad", model.parameters(), 0.4, accumulator)
            before = model.encoder(inputs).norm(dim=-1).mean().item()
            optimizer.zero_grad()
            model(inputs)[0].backward()
            optimizer.step()
            norms[accumulator] = model.encoder(inputs).norm(dim=-1).mean().item() / before
        # PyTorch's default accumulator of 0 turns the first step into lr * sign(grad).
        self.assertGreater(norms[0.0], 10.0)
        self.assertLess(norms[0.1], 2.0)

    def test_generator_records_stabilization_options(self):
        import subprocess
        import sys
        import numpy as np

        root = Path(__file__).resolve().parents[1]
        words = ["lipstick red", "shampoo mint", "serum vitamin", "mascara black",
                 "cleanser foam", "toner rose", "brush kabuki", "polish nail"]
        with TemporaryDirectory() as temporary:
            items = Path(temporary) / "items.jsonl"
            items.write_text("".join(json.dumps({"asin": "A{:02d}".format(index),
                                                 "content_text": "{} {} item {}".format(
                                                     words[index % 8], words[(index * 3) % 8], index)})
                                     + "\n" for index in range(24)))
            output = Path(temporary) / "sid"
            command = [sys.executable, "-m", "Representations.SID.scripts.generate_sids",
                       "--items", str(items), "--output-dir", str(output), "--encoder", "tfidf",
                       "--text-dim", "8", "--epochs", "2", "--batch-size", "8",
                       "--codebook-size", "4", "--hidden-dims", "8", "--latent-dim", "4",
                       "--optimizer", "adagrad", "--adagrad-initial-accumulator", "0.1",
                       "--learning-rate", "0.4", "--reset-dead-codes", "--standardize-embeddings"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIsNone(manifest["recipe"])
            self.assertEqual((manifest["optimizer"], manifest["adagrad_initial_accumulator"]),
                             ("adagrad", 0.1))
            self.assertTrue(manifest["reset_dead_codes"] and manifest["standardize_embeddings"])
            self.assertEqual(len(manifest["dead_codes_reset_total"]), 3)
            saved = np.load(output / "embeddings.npz")
            self.assertEqual(saved["standardize_mean"].shape, saved["embeddings"].shape[1:])
            rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows[-1]["dead_codes_reset"]), 3)

            rejected = subprocess.run(command[:6] + [str(output) + "-recipe", "--encoder", "tfidf",
                                                     "--recipe", "stable"],
                                      cwd=root, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--recipe requires the full Sentence-T5 catalog", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
