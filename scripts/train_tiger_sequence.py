"""Train a TIGER-style seq2seq model on prepared Amazon next-item examples."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from Recall.TIGER.sequence import SIDTokenizer, TIGERTransformer, constrained_beam_search


DEFAULT_DATA = ROOT / "data/amazon_reviews"
DEFAULT_RUNS = ROOT / "Recall/TIGER/runs"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_examples(path, limit, tokenizer):
    examples = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["target_asin"] not in tokenizer.item_tokens:
                raise ValueError("example target missing from SID catalog")
            if row["target_sid"] != tokenizer.semantic_ids[row["target_asin"]]:
                raise ValueError("example target SID differs from SID catalog")
            try:
                row["history"] = [tokenizer.sid_item[tuple(sid)]
                                  for sid in row["history_sids"]]
            except KeyError as error:
                raise ValueError("history SID missing from catalog") from error
            if not row["history"]:
                raise ValueError("example history empty")
            examples.append(row)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError("no examples in {}".format(path))
    return examples


def collate(examples, tokenizer, max_history):
    source_sequences = [tokenizer.history_tokens(row["history"], max_history)
                        for row in examples]
    max_length = max(map(len, source_sequences))
    source = torch.zeros((len(examples), max_length), dtype=torch.long)
    for index, sequence in enumerate(source_sequences):
        source[index, :len(sequence)] = torch.tensor(sequence)
    labels = torch.tensor([tokenizer.target_tokens(row["target_asin"])
                           for row in examples], dtype=torch.long)
    decoder_input = torch.cat((torch.full((len(examples), 1), tokenizer.BOS),
                               labels[:, :-1]), dim=1)
    return source, decoder_input, labels


def evaluate(model, tokenizer, examples, max_history, device, beam_size):
    model.eval()
    hits = {5: 0, 10: 0}
    ndcg = {5: 0.0, 10: 0.0}
    for row in examples:
        recommendations = constrained_beam_search(model, tokenizer, row["history"],
                                                  max_history, beam_size, device)
        for k in (5, 10):
            if row["target_asin"] in recommendations[:k]:
                rank = recommendations.index(row["target_asin"]) + 1
                hits[k] += 1
                ndcg[k] += 1 / math.log2(rank + 1)
    return {**{"recall@{}".format(k): hits[k] / len(examples) for k in (5, 10)},
            **{"ndcg@{}".format(k): ndcg[k] / len(examples) for k in (5, 10)},
            "validation_examples": len(examples), "beam_size": beam_size,
            "decoding": "catalog-constrained beam search"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-run", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--sid-method", choices=("sentence-t5_rqvae", "sentence-t5_kmeans",
                                                 "sentence-t5_lsh"), default="sentence-t5_rqvae")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if (not args.run_name or Path(args.run_name).name != args.run_name
            or args.run_name in (".", "..")):
        parser.error("--run-name must be one directory name")
    if min(args.epochs, args.batch_size, args.max_history, args.beam_size,
           args.d_model, args.heads, args.layers, args.feedforward_dim) < 1:
        parser.error("training dimensions and counts must be positive")
    for value in (args.max_train_examples, args.max_validation_examples):
        if value is not None and value < 1:
            parser.error("example limits must be positive")
    if args.beam_size < 10:
        parser.error("--beam-size must be at least 10 to report Recall@10")
    run_dir = args.runs_dir / args.run_name
    if run_dir.exists():
        parser.error("run directory already exists: {}".format(run_dir))
    sid_path = args.sid_run / "semantic_ids.jsonl"
    manifest_path = args.sid_run / "manifest.json"
    if not sid_path.is_file() or not manifest_path.is_file():
        parser.error("SID run is missing semantic_ids.jsonl or manifest.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    with manifest_path.open(encoding="utf-8") as source:
        sid_manifest = json.load(source)
    with sid_path.open(encoding="utf-8") as source:
        sids = {row["asin"]: row["sid"] for row in map(json.loads, source)}
    if len(sids) != sid_manifest["item_count"]:
        parser.error("SID catalog count differs from manifest")
    tokenizer = SIDTokenizer(sids, sid_manifest["codebook_size"])
    item_path = args.data_root / "2014/train/raw" / args.category / "items.jsonl"
    if not item_path.is_file() or file_sha256(item_path) != sid_manifest["source_sha256"]:
        parser.error("SID catalog does not match this category's items.jsonl")
    sid_data_root = args.data_root / "2014/train/sids" / args.sid_method / args.category
    sid_split_manifest_path = sid_data_root / "split_manifest.json"
    if not sid_split_manifest_path.is_file():
        parser.error("materialize SID splits first: {}".format(sid_split_manifest_path))
    with sid_split_manifest_path.open(encoding="utf-8") as source:
        sid_split_manifest = json.load(source)
    if (sid_split_manifest["method"] != args.sid_method
            or sid_split_manifest["sid_sha256"] != file_sha256(sid_path)
            or sid_split_manifest["items_sha256"] != file_sha256(item_path)):
        parser.error("materialized SID split does not match this SID run or item catalog")
    train_path = sid_data_root / "examples.jsonl"
    eval_path = (args.data_root / "2014/validation/sids" / args.sid_method
                 / args.category / "examples.jsonl")
    train = read_examples(train_path, args.max_train_examples, tokenizer)
    evaluation = read_examples(eval_path, args.max_validation_examples, tokenizer)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = TIGERTransformer(tokenizer.vocab_size, args.max_history,
                             args.d_model, args.heads, args.layers,
                             args.feedforward_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer, args.max_history))
    print("Training on {} examples; validating {} examples".format(len(train), len(evaluation)), flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for source, decoder_input, labels in loader:
            source, decoder_input, labels = (tensor.to(args.device)
                                             for tensor in (source, decoder_input, labels))
            optimizer.zero_grad()
            logits = model(source, decoder_input)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(source)
        print("epoch {}/{} loss={:.6f}".format(epoch + 1, args.epochs,
                                              total_loss / len(train)), flush=True)
    metrics = evaluate(model, tokenizer, evaluation, args.max_history,
                       args.device, args.beam_size)
    run_dir.mkdir(parents=True)
    torch.save(model.state_dict(), run_dir / "sequence.pt")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
        output.write("\n")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=False)
    config = {"sid_run": str(args.sid_run.resolve()),
              "sid_sha256": file_sha256(sid_path),
              "train_sha256": file_sha256(train_path),
              "validation_sha256": file_sha256(eval_path),
              "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
              "category": args.category, "device": args.device,
              "sid_method": args.sid_method,
              "sid_split_manifest_sha256": file_sha256(sid_split_manifest_path),
              "decoding": "catalog-constrained beam search", "beam_size": args.beam_size,
              "train_examples": len(train), "validation_examples": len(evaluation),
              "epochs": args.epochs, "max_history": args.max_history,
              "batch_size": args.batch_size, "d_model": args.d_model,
              "heads": args.heads, "layers": args.layers,
              "feedforward_dim": args.feedforward_dim,
              "learning_rate": args.learning_rate, "seed": args.seed}
    with (run_dir / "config.json").open("w", encoding="utf-8") as output:
        json.dump(config, output, indent=2)
        output.write("\n")
    print(json.dumps(metrics), flush=True)
    print("Saved {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
