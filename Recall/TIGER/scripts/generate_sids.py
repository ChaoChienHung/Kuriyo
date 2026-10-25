"""Build Sentence-T5 item embeddings, train RQ-VAE, and write unique TIGER SIDs.

Run from the repository root with
``python3 -m Recall.TIGER.scripts.generate_sids --help``.
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

from Recall.TIGER.rqvae import RQVAE


DEFAULT_ITEMS = ROOT / "data/amazon_reviews/2014/train/raw/Beauty/items.jsonl"
DEFAULT_RUNS = ROOT / "Recall/TIGER/runs"


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
    if args.codebook_size > len(items):
        parser.error("--codebook-size cannot exceed the number of items")
    print("Encoding {} items with {}".format(len(items), args.encoder), flush=True)
    embeddings = embed_items(items, args)
    if not np.isfinite(embeddings).all():
        raise ValueError("non-finite item embedding")
    model = RQVAE(embeddings.shape[1], hidden_dims=tuple(args.hidden_dims),
                  latent_dim=args.latent_dim, codebook_size=args.codebook_size)
    initialize_codebooks(model, embeddings, args.seed)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss, _, _ = model(batch.to(args.device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print("epoch {}/{} loss={:.6f}".format(epoch + 1, args.epochs, total / len(dataset)), flush=True)

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
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "seed": args.seed, "collision_groups": collision_groups,
        "largest_collision_group": largest_collision, "codebook_usage": usages,
        "sid_length": 4, "collision_token_rule": "zero-based rank by asin within each 3-code group",
    }
    run_dir.mkdir(parents=True)
    np.savez_compressed(run_dir / "embeddings.npz", asins=np.asarray(asins), embeddings=embeddings)
    torch.save(model.state_dict(), run_dir / "rqvae.pt")
    with (run_dir / "semantic_ids.jsonl").open("w", encoding="utf-8") as output:
        for asin in asins:
            output.write(json.dumps({"asin": asin, "sid": sids[asin]}) + "\n")
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(json.dumps({"run_dir": str(run_dir), "item_count": len(items),
                      "collision_groups": collision_groups, "codebook_usage": usages}), flush=True)


if __name__ == "__main__":
    main()
