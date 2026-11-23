"""Build Sentence-T5 item embeddings, train RQ-VAE, and write unique TIGER SIDs.

Run from the repository root with
``python3 -m Representations.SID.scripts.generate_sids --help``.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from Representations.SID.rqvae import RQVAE


from core.run_logging import RunLogger


DEFAULT_ITEMS = ROOT / "data/amazon_reviews/2014/catalog/Beauty/items.jsonl"
DEFAULT_RUNS = ROOT / "Representations/SID/runs"

# Full-budget recipes share the paper's architecture and schedule (three
# 256-code levels, 32-d latent, 512/256/128 MLP, batch 1024, 20k epochs,
# k-means init from 1,024 items) and differ only in how training is stabilized.
# Each produces a distinct SID artifact; record which one in SID_VERSION.
RECIPES = {
    # PyTorch Adagrad as-is (accumulator 0). Reproduces job 1153307; the first
    # step moves every weight by ~lr and collapses the codebooks.
    "paper-pytorch": {"optimizer": "adagrad", "learning_rate": 0.4,
                      "adagrad_initial_accumulator": 0.0,
                      "reset_dead_codes": False, "standardize_embeddings": False},
    # Adagrad with the TF/Optax default accumulator plus dead-code resets.
    "paper-tf-adagrad": {"optimizer": "adagrad", "learning_rate": 0.4,
                         "adagrad_initial_accumulator": 0.1,
                         "reset_dead_codes": True, "standardize_embeddings": False},
    # AdamW on per-dimension standardized embeddings plus dead-code resets;
    # the only local diagnostic setting whose decoder learned beyond the mean.
    "stable": {"optimizer": "adamw", "learning_rate": 0.001,
               "adagrad_initial_accumulator": None,
               "reset_dead_codes": True, "standardize_embeddings": True},
}


def read_items(path, limit):
    items = []
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            if limit is None or len(items) < limit:
                record = json.loads(line)
                if not record.get("asin"):
                    raise ValueError("item without asin")
                items.append(record)
    if len(items) < 2:
        raise ValueError("at least two items are required")
    asins = [item["asin"] for item in items]
    if len(asins) != len(set(asins)):
        raise ValueError("duplicate asin in catalog")
    return items, digest.hexdigest()


def apply_texts(items, path):
    """Replace each item's content_text with the one in an item-text file (scripts/build_item_texts.py).

    Returns the manifest's text record. Every item must be covered; extra rows are refused."""
    texts = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["asin"] in texts:
                raise ValueError("duplicate asin in item texts: {}".format(row["asin"]))
            texts[row["asin"]] = row["content_text"]
    asins = [item["asin"] for item in items]
    if set(texts) != set(asins):
        raise ValueError("item texts do not cover exactly the catalog items")
    for item in items:
        item["content_text"] = texts[item["asin"]]
    sidecar = path.with_name(path.name.replace(".jsonl", ".manifest.json"))
    recipe = json.loads(sidecar.read_text()).get("recipe") if sidecar.is_file() else None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"recipe": recipe or path.stem, "path": str(path.resolve()), "sha256": digest}


def embed_items(items, args):
    texts = [item.get("content_text") or item.get("title") or "unknown item"
             for item in items]
    if args.encoder == "sentence-t5":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.text_model, device=args.device)
        embeddings = encoder.encode(texts, batch_size=args.text_batch_size,
                                    convert_to_numpy=True, normalize_embeddings=True,
                                    show_progress_bar=True)
        return np.asarray(embeddings, dtype=np.float32)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    features = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2),
                               stop_words="english").fit_transform(texts)
    dimensions = min(args.text_dim, features.shape[0] - 1, features.shape[1] - 1)
    if dimensions < 1:
        raise ValueError("too few distinct text features for TF-IDF/SVD")
    embeddings = TruncatedSVD(n_components=dimensions, random_state=args.seed).fit_transform(features)
    return normalize(embeddings).astype(np.float32)


def initialize_codebooks(model, embeddings, seed):
    """Initialize each codebook with k-means on successive latent residuals."""
    from sklearn.cluster import MiniBatchKMeans
    model.eval()
    with torch.no_grad():
        latent = model.encoder(torch.from_numpy(embeddings)).numpy()
    residual = latent
    for level, codebook in enumerate(model.codebooks):
        kmeans = MiniBatchKMeans(n_clusters=codebook.shape[0], random_state=seed + level,
                                 batch_size=min(1024, len(residual)), n_init=1)
        indices = kmeans.fit_predict(residual)
        centers = kmeans.cluster_centers_.astype(np.float32)
        codebook.data.copy_(torch.from_numpy(centers))
        residual = residual - centers[indices]


def standardize(embeddings):
    """Per-dimension z-score; Sentence-T5 vectors share a large common component."""
    mean = embeddings.mean(0)
    std = embeddings.std(0) + 1e-8
    return ((embeddings - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_optimizer(name, parameters, learning_rate, adagrad_initial_accumulator):
    if name == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate,
                                   initial_accumulator_value=adagrad_initial_accumulator)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError("unknown optimizer: {}".format(name))


def unique_semantic_ids(asins, codes):
    groups = defaultdict(list)
    for asin, code in zip(asins, codes):
        groups[tuple(int(value) for value in code)].append(asin)
    result = {}
    for code, members in groups.items():
        for collision_token, asin in enumerate(sorted(members)):
            result[asin] = [*code, collision_token]
    assert len(result) == len(asins)
    assert len({tuple(sid) for sid in result.values()}) == len(asins)
    return result, sum(len(members) > 1 for members in groups.values()), max(map(len, groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path,
                        help="write SID artifacts here instead of a named runs/ directory")
    parser.add_argument("--encoder", choices=("sentence-t5", "tfidf"), default="sentence-t5")
    parser.add_argument("--recipe", choices=sorted(RECIPES),
                        help="full-budget SID recipe (paper architecture, 20k epochs); see RECIPES")
    parser.add_argument("--paper-baseline", action="store_true",
                        help="alias for --recipe paper-pytorch (kept for older commands)")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "adagrad"), default="adam",
                        help="optimizer for non-recipe runs")
    parser.add_argument("--adagrad-initial-accumulator", type=float, default=0.0)
    parser.add_argument("--reset-dead-codes", action="store_true",
                        help="after each epoch, re-seed codewords no item selects")
    parser.add_argument("--standardize-embeddings", action="store_true",
                        help="z-score each embedding dimension before training")
    parser.add_argument("--texts", type=Path,
                        help="item-text file from scripts.build_item_texts (asin, content_text); "
                             "default uses the catalog's content_text (title, brand, price, categories)")
    parser.add_argument("--text-model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(512, 256, 128))
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if args.paper_baseline:
        if args.recipe not in (None, "paper-pytorch"):
            parser.error("--paper-baseline is an alias for --recipe paper-pytorch")
        args.recipe = "paper-pytorch"
    if args.recipe is not None:
        if args.encoder != "sentence-t5" or args.limit_items is not None:
            parser.error("--recipe requires the full Sentence-T5 catalog")
        args.epochs = 20000
        args.batch_size = 1024
        args.codebook_size = 256
        args.hidden_dims = (512, 256, 128)
        args.latent_dim = 32
        for key, value in RECIPES[args.recipe].items():
            setattr(args, key, value)
    if args.optimizer == "adagrad" and args.adagrad_initial_accumulator < 0:
        parser.error("--adagrad-initial-accumulator must be nonnegative")
    if args.optimizer != "adagrad":
        args.adagrad_initial_accumulator = None
    if args.output_dir is None and (not args.run_name
            or Path(args.run_name).name != args.run_name
            or args.run_name in (".", "..")):
        parser.error("provide --output-dir or a one-directory --run-name")
    if args.output_dir is not None and args.run_name is not None:
        parser.error("choose --output-dir or --run-name, not both")
    if min(args.epochs, args.batch_size, args.codebook_size, args.latent_dim,
           args.text_batch_size, *args.hidden_dims) < 1:
        parser.error("training dimensions and counts must be positive")
    if args.codebook_size < 2:
        parser.error("--codebook-size must be at least 2")
    if args.limit_items is not None and args.limit_items < 2:
        parser.error("--limit-items must be at least 2")
    if not args.items.is_file():
        parser.error("items file missing: {}".format(args.items))
    run_dir = args.output_dir if args.output_dir is not None else args.runs_dir / args.run_name
    if run_dir.exists():
        parser.error("run directory already exists: {}".format(run_dir))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    items, items_sha256 = read_items(args.items, args.limit_items)
    text_source = {"recipe": "catalog", "path": None, "sha256": None}
    if args.texts is not None:
        if args.limit_items is not None:
            parser.error("--texts needs the full catalog")
        try:
            text_source = apply_texts(items, args.texts)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
    if args.codebook_size > len(items):
        parser.error("--codebook-size cannot exceed the number of items")
    run_dir.mkdir(parents=True)
    with RunLogger(run_dir) as logger:
        logger.event("sid_started", item_count=len(items), epochs=args.epochs)
        print("Encoding {} items with {}".format(len(items), args.encoder), flush=True)
        embeddings = embed_items(items, args)
        if not np.isfinite(embeddings).all():
            raise ValueError("non-finite item embedding")
        standardization = {}
        inputs = embeddings
        if args.standardize_embeddings:
            inputs, mean, std = standardize(embeddings)
            standardization = {"standardize_mean": mean, "standardize_std": std}
        model = RQVAE(inputs.shape[1], hidden_dims=tuple(args.hidden_dims),
                      latent_dim=args.latent_dim, codebook_size=args.codebook_size)
        if args.recipe is not None:
            initial_indices = np.random.default_rng(args.seed).permutation(len(inputs))[:1024]
            initialize_codebooks(model, inputs[initial_indices], args.seed)
        else:
            initialize_codebooks(model, inputs, args.seed)
        model.to(args.device)
        optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate,
                                    args.adagrad_initial_accumulator)
        dataset = TensorDataset(torch.from_numpy(inputs))
        all_inputs = torch.from_numpy(inputs).to(args.device)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        reset_generator = torch.Generator().manual_seed(args.seed)
        total_resets = [0] * len(model.codebooks)
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                loss, _, _ = model(batch.to(args.device))
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            record = {"epoch": epoch + 1, "loss": total / len(dataset),
                      "learning_rate": args.learning_rate}
            if args.reset_dead_codes:
                resets = model.reset_dead_codes(all_inputs, reset_generator)
                total_resets = [old + new for old, new in zip(total_resets, resets)]
                record["dead_codes_reset"] = resets
            if not np.isfinite(record["loss"]):
                raise RuntimeError("non-finite RQ-VAE loss at epoch {}".format(epoch + 1))
            logger.train(**record)
            if args.recipe is None or epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
                print("epoch {}/{} loss={:.6f}{}".format(
                    epoch + 1, args.epochs, record["loss"],
                    " dead_codes_reset={}".format(record["dead_codes_reset"])
                    if args.reset_dead_codes else ""), flush=True)

        model.eval()
        codes = []
        with torch.no_grad():
            for (batch,) in DataLoader(dataset, batch_size=args.batch_size):
                codes.extend(model.encode_codes(batch.to(args.device)).cpu().tolist())
        asins = [item["asin"] for item in items]
        sids, collision_groups, largest_collision = unique_semantic_ids(asins, codes)
        usages = [len(set(code[level] for code in codes)) for level in range(3)]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=False)
        manifest = {
            "method": "{}_rqvae".format(args.encoder),
            "source_items": str(args.items.resolve()), "source_sha256": items_sha256,
            "item_count": len(items), "encoder": args.encoder,
            "text_model": args.text_model if args.encoder == "sentence-t5" else None,
            "text_dim": args.text_dim if args.encoder == "tfidf" else None,
            "max_features": args.max_features if args.encoder == "tfidf" else None,
            "limit_items": args.limit_items, "text_batch_size": args.text_batch_size,
            "embedding_dim": int(embeddings.shape[1]), "epochs": args.epochs,
            "codebook_size": args.codebook_size, "latent_dim": args.latent_dim,
            "hidden_dims": args.hidden_dims, "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "device": args.device,
            "recipe": args.recipe, "paper_baseline": args.recipe == "paper-pytorch",
            "optimizer": args.optimizer,
            "adagrad_initial_accumulator": args.adagrad_initial_accumulator,
            "reset_dead_codes": args.reset_dead_codes,
            "dead_codes_reset_total": total_resets if args.reset_dead_codes else None,
            "standardize_embeddings": args.standardize_embeddings,
            "codebook_usage_ratio": [usage / args.codebook_size for usage in usages],
            "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "seed": args.seed, "collision_groups": collision_groups,
            "largest_collision_group": largest_collision, "codebook_usage": usages,
            "sid_length": 4, "collision_token_rule": "zero-based rank by asin within each 3-code group",
            "text_source": text_source,
        }
        np.savez_compressed(run_dir / "embeddings.npz", asins=np.asarray(asins),
                            embeddings=embeddings, **standardization)
        torch.save(model.state_dict(), run_dir / "rqvae.pt")
        with (run_dir / "semantic_ids.jsonl").open("w", encoding="utf-8") as output:
            for asin in asins:
                output.write(json.dumps({"asin": asin, "sid": sids[asin]}) + "\n")
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
        print(json.dumps({"run_dir": str(run_dir), "item_count": len(items),
                          "collision_groups": collision_groups, "codebook_usage": usages}), flush=True)
        logger.event("sid_completed", item_count=len(items))


if __name__ == "__main__":
    main()
