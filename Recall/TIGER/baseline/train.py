"""Train a paper-configured TIGER T5 baseline on versioned Amazon Beauty SIDs.

This PyTorch/Hugging Face implementation follows reported TIGER dimensions and
schedule; it is not the authors' T5X code. Run --help from the repository root.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from transformers import Adafactor, T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from Recall.TIGER.baseline.train_sequence import read_examples
from Recall.TIGER.baseline.sequence import SIDTokenizer
from Recall.core.data import pad_token_sequences
from Recall.core.experiment import file_sha256, git_revision, seed_everything, write_json
from Recall.core.metrics import ranking_metrics
from core.run_logging import RunLogger


def user_token(user_id, offset, buckets=2000):
    """Stable 2000-bucket hashing trick; Python's hash() is process-randomized."""
    number = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest()[:8], "big")
    return offset + number % buckets


def collate(examples, tokenizer, max_history):
    source = pad_token_sequences([
        [user_token(row["user_id"], tokenizer.vocab_size),
         *tokenizer.history_tokens(row["history"], max_history)] for row in examples
    ], tokenizer.PAD)
    labels = torch.tensor([tokenizer.target_tokens(row["target_asin"])
                           for row in examples], dtype=torch.long)
    return source, source.ne(tokenizer.PAD), labels


def build_model(tokenizer):
    config = T5Config(vocab_size=tokenizer.vocab_size + 2000, d_model=128,
                      d_kv=64, d_ff=1024, num_layers=4, num_decoder_layers=4,
                      num_heads=6, dropout_rate=0.1, dense_act_fn="relu",
                      pad_token_id=tokenizer.PAD, eos_token_id=tokenizer.EOS,
                      decoder_start_token_id=tokenizer.BOS)
    return T5ForConditionalGeneration(config)


@torch.no_grad()
def evaluate(model, tokenizer, examples, max_history, device, batch_size, beam_size):
    model.eval()
    requests = []
    invalid = 0
    total_beams = 0
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        source, mask, _ = collate(rows, tokenizer, max_history)
        sequences = model.generate(input_ids=source.to(device),
                                   attention_mask=mask.to(device),
                                   num_beams=beam_size,
                                   num_return_sequences=beam_size,
                                   max_new_tokens=5,
                                   early_stopping=True)
        sequences = sequences.cpu().tolist()
        for index, row in enumerate(rows):
            items = []
            seen = set()
            for sequence in sequences[index * beam_size:(index + 1) * beam_size]:
                total_beams += 1
                tokens = tuple(sequence[1:5])
                item = tokenizer.token_item.get(tokens)
                if item is None:
                    invalid += 1
                    items.append(None)
                elif item in seen:
                    items.append(None)
                else:
                    seen.add(item)
                    items.append(item)
            requests.append((items, row["target_asin"]))
    return {**ranking_metrics(requests, ks=(5, 10)),
            "validation_examples": len(examples), "beam_size": beam_size,
            "invalid_beam_fraction": invalid / total_beams,
            "decoding": "unconstrained beam search; invalid IDs count as misses"}


def save_checkpoint(path, model, optimizer, step, sampler, permutation, cursor, best):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "step": step, "sampler_state": sampler.get_state(),
             "permutation": permutation, "cursor": cursor, "best_ndcg10": best,
             "torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-run", type=Path, required=True)
    parser.add_argument("--sid-method", default="sentence-t5_rqvae")
    parser.add_argument("--sid-version")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/amazon_reviews")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "Recall/TIGER/baseline/runs")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--adafactor-scale-parameter", action="store_true",
                        help="scale each Adafactor step by the parameter RMS, as T5X does by default "
                             "(multiply_by_parameter_scale); without it every step is about lr in "
                             "absolute size, which is the behaviour of runs before this option")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-test-only", action="store_true",
                        help="evaluate the selected best checkpoint on held-out test once")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("--run-name must be one directory name")
    sid_version = args.sid_version or args.sid_run.name
    if Path(sid_version).name != sid_version or sid_version in ("", ".", ".."):
        parser.error("--sid-version must be one directory name")
    if min(args.steps, args.batch_size, args.eval_batch_size, args.eval_every,
           args.save_every, args.beam_size, args.max_history) < 1:
        parser.error("counts must be positive")
    if args.beam_size < 10:
        parser.error("--beam-size must be at least 10")
    for limit in (args.max_train_examples, args.max_validation_examples):
        if limit is not None and limit < 1:
            parser.error("example limits must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    sid_path = args.sid_run / "semantic_ids.jsonl"
    sid_manifest_path = args.sid_run / "manifest.json"
    if not sid_path.is_file() or not sid_manifest_path.is_file():
        parser.error("SID run is incomplete")
    sid_manifest = json.loads(sid_manifest_path.read_text())
    if sid_manifest.get("method") != args.sid_method:
        parser.error("SID method differs from --sid-method")
    if sid_manifest.get("limit_items") is not None:
        parser.error("limited-item SID run cannot train the full baseline")
    with sid_path.open(encoding="utf-8") as source:
        sids = {row["asin"]: row["sid"] for row in map(json.loads, source)}
    if len(sids) != sid_manifest["item_count"]:
        parser.error("SID count differs from producer manifest")
    tokenizer = SIDTokenizer(sids, sid_manifest["codebook_size"])
    year = args.data_root / "2014"
    item_path = year / "catalog" / args.category / "items.jsonl"
    if not item_path.is_file() or file_sha256(item_path) != sid_manifest["source_sha256"]:
        parser.error("SID producer does not match item catalog")
    split_root = year / "train/sids" / args.sid_method / args.category / sid_version
    split_manifest_path = split_root / "split_manifest.json"
    if not split_manifest_path.is_file():
        parser.error("materialize SID examples first")
    split_manifest = json.loads(split_manifest_path.read_text())
    if (split_manifest["sid_sha256"] != file_sha256(sid_path)
            or split_manifest["sid_version"] != sid_version
            or split_manifest["method"] != args.sid_method):
        parser.error("materialized split differs from SID run")
    train_path = split_root / "examples.jsonl"
    validation_path = year / "validation/sids" / args.sid_method / args.category / sid_version / "examples.jsonl"
    train = read_examples(train_path, args.max_train_examples, tokenizer)
    validation = read_examples(validation_path, args.max_validation_examples, tokenizer)
    seed_everything(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = build_model(tokenizer).to(args.device)
    optimizer = Adafactor(model.parameters(), lr=0.01, scale_parameter=args.adafactor_scale_parameter,
                          relative_step=False, warmup_init=False)
    run_dir = args.runs_dir / args.run_name
    config = {"implementation": "Hugging Face T5 PyTorch; paper-configured, not T5X",
              "sid_run": str(args.sid_run.resolve()), "sid_sha256": file_sha256(sid_path),
              "sid_method": args.sid_method, "sid_version": sid_version,
              "train_sha256": file_sha256(train_path),
              "validation_sha256": file_sha256(validation_path),
              "split_manifest_sha256": file_sha256(split_manifest_path),
              "git_revision": git_revision(ROOT), "category": args.category,
              "steps": args.steps, "batch_size": args.batch_size,
              "eval_batch_size": args.eval_batch_size, "eval_every": args.eval_every,
              "save_every": args.save_every, "beam_size": args.beam_size,
              "max_history": args.max_history, "max_train_examples": args.max_train_examples,
              "max_validation_examples": args.max_validation_examples,
              "optimizer": "Adafactor", "learning_rate": "0.01 through 10k steps; then inverse sqrt",
              "adafactor_scale_parameter": args.adafactor_scale_parameter,
              "user_hash_buckets": 2000, "seed": args.seed, "device": args.device,
              "model_config": model.config.to_dict()}
    if args.eval_test_only:
        best_path = run_dir / "best.pt"
        test_metrics_path = run_dir / "test-metrics.json"
        if not best_path.is_file() or test_metrics_path.exists():
            parser.error("best checkpoint missing or test already evaluated")
        previous = json.loads((run_dir / "config.json").read_text())
        for key in ("sid_sha256", "train_sha256", "validation_sha256", "sid_version"):
            if previous[key] != config[key]:
                parser.error("test dataset differs from selected run: {}".format(key))
        test_path = year / "test/sids" / args.sid_method / args.category / sid_version / "examples.jsonl"
        test = read_examples(test_path, None, tokenizer)
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        with RunLogger(run_dir) as logger:
            metrics = evaluate(model, tokenizer, test, args.max_history, args.device,
                               args.eval_batch_size, args.beam_size)
            metrics["test_examples"] = metrics.pop("validation_examples")
            metrics["selected_validation_ndcg@10"] = json.loads(
                (run_dir / "best-metrics.json").read_text())["ndcg@10"]
            write_json(test_metrics_path, metrics)
            print(json.dumps(metrics), flush=True)
            logger.event("test_completed", metrics=metrics)
        return
    if args.resume:
        if not (run_dir / "latest.pt").is_file():
            parser.error("--resume requires latest.pt")
        previous = json.loads((run_dir / "config.json").read_text())
        for key in ("sid_sha256", "train_sha256", "validation_sha256", "seed", "batch_size"):
            if previous[key] != config[key]:
                parser.error("resume config mismatch: {}".format(key))
        # Runs written before the option existed used scale_parameter=False.
        if previous.get("adafactor_scale_parameter", False) != args.adafactor_scale_parameter:
            parser.error("resume config mismatch: adafactor_scale_parameter")
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        sampler = torch.Generator()
        sampler.set_state(state["sampler_state"])
        permutation, cursor = state["permutation"], state["cursor"]
        best = state["best_ndcg10"]
        torch.set_rng_state(state["torch_rng_state"])
        if args.device == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    else:
        if run_dir.exists():
            parser.error("run directory already exists: {}".format(run_dir))
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", config)
        start_step = 0
        sampler = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(train), generator=sampler)
        cursor = 0
        best = -1.0
    with RunLogger(run_dir) as logger:
        logger.event("resume" if args.resume else "training_started", starting_step=start_step, target_steps=args.steps)
        print("T5 parameters={} train={} validation={} starting_step={}".format(
            sum(parameter.numel() for parameter in model.parameters()), len(train),
            len(validation), start_step), flush=True)
        for step in range(start_step + 1, args.steps + 1):
            if cursor + args.batch_size > len(train):
                permutation = torch.randperm(len(train), generator=sampler)
                cursor = 0
            indices = permutation[cursor:cursor + args.batch_size].tolist()
            cursor += args.batch_size
            batch = [train[index] for index in indices]
            source, mask, labels = collate(batch, tokenizer, args.max_history)
            learning_rate = 0.01 if step <= 10000 else 0.01 * math.sqrt(10000 / step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            optimizer.zero_grad()
            loss = model(input_ids=source.to(args.device), attention_mask=mask.to(args.device),
                         labels=labels.to(args.device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at step {}".format(step))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            logger.train(step=step, loss=loss.item(), learning_rate=learning_rate)
            if step == 1 or step % 100 == 0:
                print("step {}/{} loss={:.6f} lr={:.6g}".format(
                    step, args.steps, loss.item(), learning_rate), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, tokenizer, validation, args.max_history,
                                   args.device, args.eval_batch_size, args.beam_size)
                metrics["step"] = step
                write_json(run_dir / "validation-step-{}.json".format(step), metrics)
                logger.event("validation", metrics=metrics)
                print(json.dumps(metrics), flush=True)
                if metrics["ndcg@10"] > best:
                    best = metrics["ndcg@10"]
                    torch.save(model.state_dict(), run_dir / "best.pt")
                    write_json(run_dir / "best-metrics.json", metrics)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(run_dir / "latest.pt", model, optimizer, step,
                                sampler, permutation, cursor, best)
                logger.event("checkpoint_saved", step=step, path="latest.pt")
        print("Saved {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
