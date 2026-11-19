"""Train a paper-configured TIGER T5 baseline on versioned Amazon Beauty SIDs.

This PyTorch/Hugging Face implementation follows reported TIGER dimensions and
schedule; it is not the authors' T5X code. Run --help from the repository root.
Run directory layout: core/runs.py.
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
from Recall.core.data import pad_token_sequences, preprocessing_summary
from Recall.core.metrics import ranking_metrics
from core.experiment import file_sha256, seed_everything
from core.runs import (RunError, RunLayout, Selection, add_run_arguments, check_run_arguments,
                       configure_threads, display_path, eval_set_name, evaluate_test_once,
                       load_checkpoint, open_run, run_section, save_checkpoint)

# Must match to resume; keys are the flat config.json names of runs before 2026-11-19, which
# all used scale_parameter=False before the option was recorded.
RESUME_KEYS = {"sid_sha256": "data.sid.sha256", "train_sha256": "data.splits.train.sha256",
               "validation_sha256": "data.splits.validation.sha256", "seed": "run.seed",
               "batch_size": "training.batch_size",
               "adafactor_scale_parameter": ("training.adafactor_scale_parameter", False)}
TEST_KEYS = {"sid_sha256": "data.sid.sha256", "train_sha256": "data.splits.train.sha256",
             "validation_sha256": "data.splits.validation.sha256", "sid_version": "data.sid.version"}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-run", type=Path, required=True)
    parser.add_argument("--sid-method", default="sentence-t5_rqvae")
    parser.add_argument("--sid-version")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/amazon_reviews")
    parser.add_argument("--category", default="Beauty")
    add_run_arguments(parser, ROOT / "Recall/TIGER/baseline/runs")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--adafactor-scale-parameter", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="scale each Adafactor step by the parameter RMS, as T5X does by default "
                             "(multiply_by_parameter_scale; default). --no-adafactor-scale-parameter "
                             "makes every step about lr in absolute size, the behaviour of runs "
                             "before 2026-11-19; use it to resume those runs")
    args = parser.parse_args()
    error = check_run_arguments(args)
    if error:
        parser.error(error)
    sid_version = args.sid_version or args.sid_run.name
    if Path(sid_version).name != sid_version or sid_version in ("", ".", ".."):
        parser.error("--sid-version must be one directory name")
    if min(args.steps, args.batch_size, args.eval_batch_size, args.eval_every,
           args.save_every, args.beam_size, args.max_history) < 1:
        parser.error("counts must be positive")
    if args.beam_size < 10:
        parser.error("--beam-size must be at least 10")
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
    paths = {split: year / split / "sids" / args.sid_method / args.category / sid_version / "examples.jsonl"
             for split in ("train", "validation", "test")}
    train = read_examples(paths["train"], args.max_train_examples, tokenizer)
    validation = read_examples(paths["validation"], args.max_validation_examples, tokenizer)
    seed_everything(args.seed)
    configure_threads(args.device)
    model = build_model(tokenizer).to(args.device)
    optimizer = Adafactor(model.parameters(), lr=0.01, scale_parameter=args.adafactor_scale_parameter,
                          relative_step=False, warmup_init=False)
    eval_set = eval_set_name("2014", args.category)
    layout = RunLayout(args.runs_dir / args.run_name, eval_set)
    settings = {
        "run": run_section(args.run_name, "TIGER", "baseline",
                           "Hugging Face T5 PyTorch; paper-configured, not T5X", ROOT,
                           args.device, args.seed),
        "data": {"year": "2014", "category": args.category,
                 "catalog": {"path": display_path(item_path, ROOT), "sha256": file_sha256(item_path)},
                 "sid": {"run": display_path(args.sid_run, ROOT), "method": args.sid_method,
                         "version": sid_version, "sha256": file_sha256(sid_path),
                         "recipe": sid_manifest.get("recipe"),
                         "codebook_size": sid_manifest["codebook_size"],
                         "item_count": sid_manifest["item_count"]},
                 "splits": {split: {"path": display_path(path, ROOT),
                                    "sha256": file_sha256(path) if path.is_file() else None}
                            for split, path in paths.items()},
                 "sid_split_manifest": {"path": display_path(split_manifest_path, ROOT),
                                        "sha256": file_sha256(split_manifest_path)},
                 "preprocessing": preprocessing_summary(
                     year / "train/raw" / args.category / "split_manifest.json", ROOT),
                 "max_history": args.max_history, "max_train_examples": args.max_train_examples,
                 "max_validation_examples": args.max_validation_examples,
                 "training_examples": "one example per train prefix; a hashed user token, then 4 "
                                      "tokens per history item (3 SID levels + collision suffix)"},
        "model": {"config": model.config.to_dict(), "user_hash_buckets": 2000,
                  "parameters": sum(parameter.numel() for parameter in model.parameters())},
        "training": {"optimizer": "Adafactor", "adafactor_scale_parameter": args.adafactor_scale_parameter,
                     "learning_rate": "0.01 through 10k steps; then 0.01 * sqrt(10000 / step)",
                     "gradient_clip_norm": 1.0, "steps": args.steps, "batch_size": args.batch_size,
                     "save_every": args.save_every},
        "evaluation": {"set": eval_set, "every_steps": args.eval_every,
                       "batch_size": args.eval_batch_size, "beam_size": args.beam_size,
                       "selection": "best validation NDCG@10", "ks": [5, 10],
                       "decoding": "unconstrained beam search; invalid IDs count as misses"}}
    if args.eval_test_only:
        test = read_examples(paths["test"], None, tokenizer)

        def evaluate_test():
            metrics = evaluate(model, tokenizer, test, args.max_history, args.device,
                               args.eval_batch_size, args.beam_size)
            metrics["test_examples"] = metrics.pop("validation_examples")
            return metrics

        try:
            evaluate_test_once(layout, settings, TEST_KEYS, RESUME_KEYS, model, evaluate_test,
                               settings["data"]["splits"]["test"]["sha256"])
        except RunError as error:
            parser.error(str(error))
        return
    try:
        open_run(layout, settings, RESUME_KEYS, args.resume)
    except RunError as error:
        message = str(error)
        if "adafactor_scale_parameter" in message:
            message += "; pass {}".format("--no-adafactor-scale-parameter" if args.adafactor_scale_parameter
                                          else "--adafactor-scale-parameter")
        parser.error(message)
    if args.resume:
        state = load_checkpoint(layout.latest, args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        sampler = torch.Generator()
        sampler.set_state(state["sampler_state"])
        permutation, cursor = state["permutation"], state["cursor"]
        selection = Selection.from_checkpoint(state)
    else:
        start_step = 0
        sampler = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(train), generator=sampler)
        cursor = 0
        selection = Selection()
    with layout.logger() as logger:
        logger.event("resume" if args.resume else "training_started", starting_step=start_step,
                     target_steps=args.steps, git_revision=settings["run"]["git_revision"],
                     slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        print("T5 parameters={} train={} validation={} starting_step={}".format(
            settings["model"]["parameters"], len(train), len(validation), start_step), flush=True)
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
                metrics.update(split="validation", set=eval_set, step=step)
                selection.record(layout, logger, model, metrics, "step", step)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(layout.latest, {"model": model.state_dict(),
                                                "optimizer": optimizer.state_dict(), "step": step,
                                                "sampler_state": sampler.get_state(),
                                                "permutation": permutation, "cursor": cursor,
                                                **selection.state()})
                logger.event("checkpoint_saved", step=step, path="checkpoints/latest.pt")
        print("Saved {}".format(layout.root), flush=True)


if __name__ == "__main__":
    main()
