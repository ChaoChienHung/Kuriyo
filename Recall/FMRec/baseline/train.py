"""Train and evaluate the FMRec baseline on the prepared Amazon 2014 ASIN split.

Defaults follow the authors' code (FengLiu-1/FMRec, commit 1c39b17) except the
20-item history cap of Kuriyo's split. Validation selects the checkpoint;
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

from Recall.FMRec.baseline.data import collate, load_catalog, read_examples
from Recall.FMRec.baseline.model import ATTENTION_MODES, TIME_SCHEDULES, FMRec
from Recall.core.experiment import file_sha256, git_revision, seed_everything, write_json
from Recall.core.metrics import ranking_metrics
from core.run_logging import RunLogger

UPSTREAM = "https://github.com/FengLiu-1/FMRec commit 1c39b17"
MODEL_KEYS = ("hidden", "heads", "decoder1_layers", "decoder2_layers", "dropout",
              "embedding_dropout", "lambda_uncertainty", "eps", "time_schedule",
              "mode_scale", "sample_steps", "attention")
RESUME_KEYS = ("train_sha256", "validation_sha256", "catalog_sha256", "seed",
               "batch_size", "max_history", "max_train_examples", "model")


@torch.no_grad()
def evaluate(model, rows, asins, max_history, device, batch_size, seed, ks=(5, 10)):
    """Full ranking over every catalog item; noise is seeded and leaves global RNG alone."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    requests = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        sequence, _ = collate(batch, max_history)
        representation = model.sample(sequence.to(device), generator=generator)
        top = model.scores(representation).topk(min(max(ks), model.num_items), dim=-1).indices.cpu().tolist()
        requests.extend(([asins[index - 1] for index in indices], asins[row["target"] - 1])
                        for indices, row in zip(top, batch))
    return {**ranking_metrics(requests, ks=ks), "examples": len(rows),
            "ranking": "full catalog, inner product with item embeddings; history items not filtered",
            "sample_steps": model.sample_steps, "eval_seed": seed}


def save_checkpoint(path, model, optimizer, scheduler, sampler, epoch, step, best, bad_evals, stopped):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "sampler_state": sampler.get_state(),
             "epoch": epoch, "step": step, "best_ndcg10": best, "bad_evals": bad_evals,
             "stopped_early": stopped, "torch_rng_state": torch.get_rng_state()}
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
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "Recall/FMRec/baseline/runs")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=20, help="epochs between validations")
    parser.add_argument("--patience", type=int, default=5,
                        help="stop after this many validations without NDCG@10 gain; 0 disables")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-decay-epochs", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--decoder1-layers", type=int, default=2)
    parser.add_argument("--decoder2-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding-dropout", type=float, default=0.3)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--time-schedule", choices=TIME_SCHEDULES, default="mode")
    parser.add_argument("--mode-scale", type=float, default=1.0,
                        help="s of the mode time schedule; author code default 1.0")
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--attention", choices=ATTENTION_MODES, default="bidirectional")
    parser.add_argument("--alpha", type=float, default=0.2, help="weight of L_CE")
    parser.add_argument("--beta", type=float, default=0.4, help="weight of L_MSE")
    parser.add_argument("--mask-ratio", type=float, default=1.0)
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
           args.lr_decay_epochs, args.max_history, args.sample_steps) < 1 or args.patience < 0:
        parser.error("counts must be positive")
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
    asins = load_catalog(catalog_path)
    item_index = {asin: index + 1 for index, asin in enumerate(asins)}

    model_config = {key: getattr(args, key) for key in MODEL_KEYS}
    run_dir = args.runs_dir / args.run_name
    config = {"implementation": "Kuriyo PyTorch re-implementation of FMRec", "upstream": UPSTREAM,
              "git_revision": git_revision(ROOT), "category": args.category,
              "catalog_sha256": catalog_sha, "num_items": len(asins),
              "train_sha256": file_sha256(paths["train"]),
              "validation_sha256": file_sha256(paths["validation"]),
              "split_manifest_sha256": file_sha256(split_manifest_path),
              "model": model_config, "epochs": args.epochs, "batch_size": args.batch_size,
              "eval_batch_size": args.eval_batch_size, "eval_every": args.eval_every,
              "patience": args.patience, "optimizer": "Adam", "lr": args.lr,
              "weight_decay": args.weight_decay, "lr_decay_epochs": args.lr_decay_epochs,
              "lr_gamma": args.lr_gamma, "alpha": args.alpha, "beta": args.beta,
              "mask_ratio": args.mask_ratio, "max_history": args.max_history,
              "max_train_examples": args.max_train_examples,
              "max_validation_examples": args.max_validation_examples,
              "selection": "best validation NDCG@10", "seed": args.seed, "device": args.device}

    seed_everything(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = FMRec(len(asins), **model_config).to(args.device)
    config["parameters"] = sum(parameter.numel() for parameter in model.parameters())

    if args.eval_test_only:
        best_path, test_metrics_path = run_dir / "best.pt", run_dir / "test-metrics.json"
        if not best_path.is_file() or test_metrics_path.exists():
            parser.error("best checkpoint missing or test already evaluated")
        previous = json.loads((run_dir / "config.json").read_text())
        for key in ("catalog_sha256", "train_sha256", "validation_sha256", "model", "max_history"):
            if previous[key] != config[key]:
                parser.error("test settings differ from selected run: {}".format(key))
        test = read_examples(paths["test"], item_index)
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
        with RunLogger(run_dir) as logger:
            metrics = evaluate(model, test, asins, args.max_history, args.device,
                               args.eval_batch_size, args.seed)
            metrics["split"] = "test"
            metrics["test_sha256"] = file_sha256(paths["test"])
            metrics["selected_validation"] = json.loads((run_dir / "best-metrics.json").read_text())
            write_json(test_metrics_path, metrics)
            logger.event("test_completed", metrics=metrics)
            print(json.dumps(metrics), flush=True)
        return

    train = read_examples(paths["train"], item_index, args.max_train_examples)
    validation = read_examples(paths["validation"], item_index, args.max_validation_examples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_gamma)
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
        scheduler.load_state_dict(state["scheduler"])
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
        print("FMRec parameters={} items={} train={} validation={} starting_epoch={}".format(
            config["parameters"], len(asins), len(train), len(validation), start_epoch), flush=True)
        if stopped:
            print("run already stopped early; nothing to resume", flush=True)
            logger.event("already_stopped", epoch=start_epoch)
            return
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(train), generator=sampler).tolist()
            learning_rate = optimizer.param_groups[0]["lr"]
            for start in range(0, len(train), args.batch_size):
                step += 1
                sequence, target = collate([train[i] for i in permutation[start:start + args.batch_size]],
                                           args.max_history)
                optimizer.zero_grad()
                losses = model.loss(sequence.to(args.device), target.to(args.device),
                                    args.alpha, args.beta, args.mask_ratio)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError("non-finite loss at step {}".format(step))
                losses["loss"].backward()
                optimizer.step()
                values = {name: value.item() for name, value in losses.items()}
                logger.train(step=step, epoch=epoch, learning_rate=learning_rate, **values)
                if step == 1 or step % 100 == 0:
                    print("epoch {} step {} loss={loss:.5f} fm={loss_fm:.5f} ce={loss_ce:.4f} "
                          "mse={loss_mse:.6f} lr={lr:.3g}".format(epoch, step, lr=learning_rate, **values),
                          flush=True)
            scheduler.step()
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                metrics = evaluate(model, validation, asins, args.max_history, args.device,
                                   args.eval_batch_size, args.seed)
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
            save_checkpoint(run_dir / "latest.pt", model, optimizer, scheduler, sampler,
                            epoch, step, best, bad_evals, stopped)
            logger.event("checkpoint_saved", epoch=epoch, step=step, path="latest.pt")
            if stopped:
                logger.event("early_stopped", epoch=epoch, best_ndcg10=best)
                print("early stop at epoch {}".format(epoch), flush=True)
                break
        print("Saved {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
