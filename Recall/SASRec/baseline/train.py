"""Train and evaluate the SASRec baseline on the prepared Amazon 2014 ASIN split.

A sanity baseline: full-catalog ranking on the same leave-one-out split and the
same shared metrics as TIGER and FMRec. Validation selects the checkpoint;
held-out test runs only with --eval-test-only. Run --help from the repo root.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from Recall.SASRec.baseline.data import (collate_histories, collate_windows, load_catalog,
                                         read_examples, user_sequences, windows)
from Recall.SASRec.baseline.model import LOSSES, SASRec
from Recall.core.experiment import file_sha256, git_revision, seed_everything, write_json
from Recall.core.metrics import ranking_metrics
from core.run_logging import RunLogger

MODEL_KEYS = ("max_history", "hidden", "heads", "layers", "dropout")
RESUME_KEYS = ("train_sha256", "validation_sha256", "catalog_sha256", "seed", "batch_size",
               "max_train_examples", "model", "loss", "lr", "weight_decay")


@torch.no_grad()
def evaluate(model, rows, asins, max_history, device, batch_size, ks=(5, 10)):
    """Full ranking over every catalog item from the newest position."""
    model.eval()
    requests = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        sequence, _ = collate_histories(batch, max_history)
        hidden = model(sequence.to(device))[:, -1]
        top = model.scores(hidden).topk(min(max(ks), model.num_items), dim=-1).indices.cpu().tolist()
        requests.extend(([asins[index] for index in indices], asins[row["target"] - 1])
                        for indices, row in zip(top, batch))
    return {**ranking_metrics(requests, ks=ks), "examples": len(rows),
            "ranking": "full catalog, inner product with item embeddings; history items not filtered"}


def save_checkpoint(path, model, optimizer, sampler, epoch, step, best, bad_evals, stopped):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sampler_state": sampler.get_state(), "epoch": epoch, "step": step,
             "best_ndcg10": best, "bad_evals": bad_evals, "stopped_early": stopped,
             "torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(state, temporary)
    os.replace(temporary, path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/amazon_reviews")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "Recall/SASRec/baseline/runs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128, help="training windows per step")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=5, help="epochs between validations")
    parser.add_argument("--patience", type=int, default=5,
                        help="stop after this many validations without NDCG@10 gain; 0 disables")
    parser.add_argument("--loss", choices=LOSSES, default="ce",
                        help="ce: full softmax over items; bce: one sampled negative (original paper)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-test-only", action="store_true",
                        help="evaluate best.pt on held-out test once")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("--run-name must be one directory name")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_every,
           args.max_history, args.hidden, args.heads, args.layers) < 1 or args.patience < 0:
        parser.error("counts must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    for limit in (args.max_train_examples, args.max_validation_examples):
        if limit is not None and limit < 1:
            parser.error("example limits must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")

    year = args.data_root / "2014"
    catalog_path = year / "catalog" / args.category / "items.jsonl"
    split_manifest_path = year / "train/raw" / args.category / "split_manifest.json"
    paths = {split: year / split / "raw" / args.category / "examples.jsonl"
             for split in ("train", "validation", "test")}
    if not catalog_path.is_file() or not split_manifest_path.is_file() or not all(
            path.is_file() for path in paths.values()):
        parser.error("prepare the 2014 {} split first (scripts.prepare_amazon_reviews_2014)".format(args.category))
    catalog_sha = file_sha256(catalog_path)
    split_manifest = json.loads(split_manifest_path.read_text())
    if split_manifest.get("catalog", {}).get("sha256") not in (None, catalog_sha):
        parser.error("split manifest was produced from a different catalog")
    split_history = split_manifest.get("rules", {}).get("max_history")
    if split_history is not None and args.max_history > split_history:
        parser.error("--max-history exceeds the split's history cap {}".format(split_history))
    asins = load_catalog(catalog_path)
    item_index = {asin: index + 1 for index, asin in enumerate(asins)}

    model_config = {key: getattr(args, key) for key in MODEL_KEYS}
    run_dir = args.runs_dir / args.run_name
    config = {"implementation": "Kuriyo PyTorch SASRec (Kang & McAuley 2018)",
              "git_revision": git_revision(ROOT), "category": args.category,
              "catalog_sha256": catalog_sha, "num_items": len(asins),
              "train_sha256": file_sha256(paths["train"]),
              "validation_sha256": file_sha256(paths["validation"]),
              "split_manifest_sha256": file_sha256(split_manifest_path),
              "model": model_config, "loss": args.loss, "epochs": args.epochs,
              "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
              "eval_every": args.eval_every, "patience": args.patience,
              "optimizer": "Adam(betas=(0.9, 0.98))", "lr": args.lr,
              "weight_decay": args.weight_decay,
              "training_data": "per-user train sequences rebuilt from prefix examples, split into "
                               "windows of max_history + 1; loss at every real position",
              "max_train_examples": args.max_train_examples,
              "max_validation_examples": args.max_validation_examples,
              "selection": "best validation NDCG@10", "seed": args.seed, "device": args.device}

    seed_everything(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = SASRec(len(asins), **model_config).to(args.device)
    config["parameters"] = sum(parameter.numel() for parameter in model.parameters())

    if args.eval_test_only:
        best_path, test_metrics_path = run_dir / "best.pt", run_dir / "test-metrics.json"
        if not best_path.is_file() or test_metrics_path.exists():
            parser.error("best checkpoint missing or test already evaluated")
        previous = json.loads((run_dir / "config.json").read_text())
        for key in ("catalog_sha256", "train_sha256", "validation_sha256", "model"):
            if previous[key] != config[key]:
                parser.error("test settings differ from selected run: {}".format(key))
        test = read_examples(paths["test"], item_index)
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
        with RunLogger(run_dir) as logger:
            metrics = evaluate(model, test, asins, args.max_history, args.device, args.eval_batch_size)
            metrics["split"] = "test"
            metrics["test_sha256"] = file_sha256(paths["test"])
            metrics["selected_validation"] = json.loads((run_dir / "best-metrics.json").read_text())
            write_json(test_metrics_path, metrics)
            logger.event("test_completed", metrics=metrics)
            print(json.dumps(metrics), flush=True)
        return

    train_rows = read_examples(paths["train"], item_index, args.max_train_examples)
    train = [window for sequence in user_sequences(train_rows, args.max_history)
             for window in windows(sequence, args.max_history)]
    validation = read_examples(paths["validation"], item_index, args.max_validation_examples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                                 weight_decay=args.weight_decay)
    if args.resume:
        if not (run_dir / "latest.pt").is_file():
            parser.error("--resume requires latest.pt")
        previous = json.loads((run_dir / "config.json").read_text())
        for key in RESUME_KEYS:
            if previous[key] != config[key]:
                parser.error("resume config mismatch: {}".format(key))
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler = torch.Generator()
        sampler.set_state(state["sampler_state"])
        start_epoch, step = state["epoch"], state["step"]
        best, bad_evals, stopped = state["best_ndcg10"], state["bad_evals"], state["stopped_early"]
        torch.set_rng_state(state["torch_rng_state"])
        if args.device == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    else:
        if run_dir.exists():
            parser.error("run directory already exists: {}".format(run_dir))
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", config)
        sampler = torch.Generator().manual_seed(args.seed)
        start_epoch, step, best, bad_evals, stopped = 0, 0, -1.0, 0, False

    with RunLogger(run_dir) as logger:
        logger.event("resume" if args.resume else "training_started", starting_epoch=start_epoch,
                     starting_step=step, target_epochs=args.epochs, git_revision=config["git_revision"],
                     slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        print("SASRec parameters={} items={} train_examples={} windows={} validation={} "
              "starting_epoch={}".format(config["parameters"], len(asins), len(train_rows), len(train),
                                         len(validation), start_epoch), flush=True)
        if stopped:
            print("run already stopped early; nothing to resume", flush=True)
            logger.event("already_stopped", epoch=start_epoch)
            return
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(train), generator=sampler).tolist()
            for start in range(0, len(train), args.batch_size):
                step += 1
                inputs, targets = collate_windows([train[i] for i in permutation[start:start + args.batch_size]],
                                                  args.max_history)
                optimizer.zero_grad()
                loss = model.loss(inputs.to(args.device), targets.to(args.device), args.loss, sampler)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss at step {}".format(step))
                loss.backward()
                optimizer.step()
                logger.train(step=step, epoch=epoch, loss=loss.item(), learning_rate=args.lr)
                if step == 1 or step % 100 == 0:
                    print("epoch {} step {} loss={:.5f}".format(epoch, step, loss.item()), flush=True)
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                metrics = evaluate(model, validation, asins, args.max_history, args.device,
                                   args.eval_batch_size)
                metrics.update(split="validation", epoch=epoch, step=step)
                write_json(run_dir / "validation-epoch-{}.json".format(epoch), metrics)
                logger.event("validation", metrics=metrics)
                print(json.dumps(metrics), flush=True)
                if metrics["ndcg@10"] > best:
                    best, bad_evals = metrics["ndcg@10"], 0
                    torch.save(model.state_dict(), run_dir / "best.pt")
                    write_json(run_dir / "best-metrics.json", metrics)
                else:
                    bad_evals += 1
                stopped = bool(args.patience) and bad_evals >= args.patience
            save_checkpoint(run_dir / "latest.pt", model, optimizer, sampler,
                            epoch, step, best, bad_evals, stopped)
            logger.event("checkpoint_saved", epoch=epoch, step=step, path="latest.pt")
            if stopped:
                logger.event("early_stopped", epoch=epoch, best_ndcg10=best)
                print("early stop at epoch {}".format(epoch), flush=True)
                break
        print("Saved {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
