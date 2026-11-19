"""Train and evaluate the FMRec baseline on the prepared Amazon 2014 ASIN split.

Defaults follow the authors' code (FengLiu-1/FMRec, commit 1c39b17) except the
20-item history cap of Kuriyo's split. Validation selects the checkpoint;
held-out test runs only with --eval-test-only. Run --help from the repo root.
Run directory layout: core/runs.py.
"""

import argparse
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from Recall.FMRec.baseline.data import collate
from Recall.FMRec.baseline.model import ATTENTION_MODES, TIME_SCHEDULES, FMRec
from Recall.core.data import AsinSplit
from Recall.core.evaluation import rank_full_catalog
from Recall.core.popularity import GROUPING, popularity_groups, target_counts
from core.experiment import seed_everything
from core.runs import (RunError, RunLayout, Selection, add_run_arguments, check_run_arguments,
                       configure_threads, evaluate_groups, evaluate_test_once, load_checkpoint, open_run,
                       run_section, save_checkpoint)

UPSTREAM = "https://github.com/FengLiu-1/FMRec commit 1c39b17"
MODEL_KEYS = ("hidden", "heads", "decoder1_layers", "decoder2_layers", "dropout",
              "embedding_dropout", "lambda_uncertainty", "eps", "time_schedule",
              "mode_scale", "sample_steps", "attention")
# Must match to resume; keys are the flat config.json names of runs before 2026-11-19.
RESUME_KEYS = {"train_sha256": "data.splits.train.sha256",
               "validation_sha256": "data.splits.validation.sha256",
               "catalog_sha256": "data.catalog.sha256", "seed": "run.seed",
               "batch_size": "training.batch_size", "max_history": "data.max_history",
               "max_train_examples": "data.max_train_examples", "model": "model.config"}
TEST_KEYS = {key: RESUME_KEYS[key] for key in
             ("catalog_sha256", "train_sha256", "validation_sha256", "model", "max_history")}


def evaluate(model, rows, asins, max_history, device, batch_size, seed, ks=(5, 10), groups=None):
    """Full ranking over every catalog item; noise is seeded and leaves global RNG alone."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)

    def score(batch):
        sequence, _ = collate(batch, max_history)
        return model.scores(model.sample(sequence.to(device), generator=generator))

    with torch.no_grad():
        metrics = rank_full_catalog(rows, asins, score, batch_size, model.num_items, ks, index_offset=1,
                                    groups=groups)
    return {**metrics, "sample_steps": model.sample_steps, "eval_seed": seed}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/amazon_reviews")
    parser.add_argument("--category", default="Beauty")
    add_run_arguments(parser, ROOT / "Recall/FMRec/baseline/runs")
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    error = check_run_arguments(args)
    if error:
        parser.error(error)
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_every,
           args.lr_decay_epochs, args.max_history, args.sample_steps) < 1 or args.patience < 0:
        parser.error("counts must be positive")
    try:
        split = AsinSplit(args.data_root, args.category)
        split.check_history(args.max_history)
    except RunError as error:
        parser.error(str(error))
    asins = split.asins

    model_config = {key: getattr(args, key) for key in MODEL_KEYS}
    layout = RunLayout(args.runs_dir / args.run_name, split.eval_set)
    settings = {
        "run": run_section(args.run_name, "FMRec", "baseline",
                           "Kuriyo PyTorch re-implementation of FMRec ({})".format(UPSTREAM),
                           ROOT, args.device, args.seed),
        "data": {**split.settings(ROOT, args.max_history, args.max_train_examples,
                                  args.max_validation_examples),
                 "training_examples": "one example per train prefix; the newest max_history items"},
        "model": {"config": model_config},
        "training": {"optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
                     "lr_schedule": "StepLR every {} epochs x{}".format(args.lr_decay_epochs, args.lr_gamma),
                     "lr_decay_epochs": args.lr_decay_epochs, "lr_gamma": args.lr_gamma,
                     "alpha": args.alpha, "beta": args.beta, "mask_ratio": args.mask_ratio,
                     "epochs": args.epochs, "batch_size": args.batch_size},
        "evaluation": {"set": split.eval_set, "every_epochs": args.eval_every,
                       "batch_size": args.eval_batch_size, "patience": args.patience,
                       "selection": "best validation NDCG@10", "ks": [5, 10],
                       "ranking": "full catalog, inner product with item embeddings; "
                                  "history items not filtered",
                       "sampling": "{} Euler steps, noise seeded with the run seed".format(args.sample_steps),
                       "groups": GROUPING}}

    seed_everything(args.seed)
    configure_threads(args.device)
    model = FMRec(len(asins), **model_config).to(args.device)
    settings["model"]["parameters"] = sum(parameter.numel() for parameter in model.parameters())

    train = split.read("train", args.max_train_examples)
    counts = target_counts(row["target"] for row in train)

    def evaluate_rows(rows):
        return evaluate(model, rows, asins, args.max_history, args.device, args.eval_batch_size,
                        args.seed, groups=popularity_groups([row["target"] for row in rows], counts))

    if args.eval_test_only or args.eval_groups:
        rows = {"validation": split.read("validation", args.max_validation_examples),
                "test": split.read("test")}
        try:
            if args.eval_test_only:
                evaluate_test_once(layout, settings, TEST_KEYS, RESUME_KEYS, model,
                                   lambda: evaluate_rows(rows["test"]), split.sha256["test"])
            else:
                evaluate_groups(layout, settings, TEST_KEYS, RESUME_KEYS, model,
                                lambda name: evaluate_rows(rows[name]))
        except RunError as error:
            parser.error(str(error))
        return

    validation = split.read("validation", args.max_validation_examples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_gamma)
    try:
        recorded = open_run(layout, settings, RESUME_KEYS, args.resume)
    except RunError as error:
        parser.error(str(error))
    if args.resume:
        state = load_checkpoint(layout.latest, args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler = torch.Generator()
        sampler.set_state(state["sampler_state"])
        start_epoch, step = state["epoch"], state["step"]
        selection = Selection.from_checkpoint(state, args.patience)
    else:
        sampler = torch.Generator().manual_seed(args.seed)
        start_epoch, step, selection = 0, 0, Selection(args.patience)

    with layout.logger() as logger:
        logger.event("resume" if args.resume else "training_started", starting_epoch=start_epoch,
                     starting_step=step, target_epochs=args.epochs,
                     git_revision=settings["run"]["git_revision"],
                     settings_git_revision=recorded["run"]["git_revision"],
                     slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        print("FMRec parameters={} items={} train={} validation={} starting_epoch={}".format(
            settings["model"]["parameters"], len(asins), len(train), len(validation), start_epoch),
            flush=True)
        if selection.stopped:
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
                metrics = evaluate_rows(validation)
                metrics.update(split="validation", set=layout.eval_set, epoch=epoch, step=step)
                selection.record(layout, logger, model, metrics, "epoch", epoch)
            save_checkpoint(layout.latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                            "scheduler": scheduler.state_dict(),
                                            "sampler_state": sampler.get_state(), "epoch": epoch,
                                            "step": step, **selection.state()})
            logger.event("checkpoint_saved", epoch=epoch, step=step, path="checkpoints/latest.pt")
            if selection.stopped:
                logger.event("early_stopped", epoch=epoch, best_ndcg10=selection.best)
                print("early stop at epoch {}".format(epoch), flush=True)
                break
        print("Saved {}".format(layout.root), flush=True)


if __name__ == "__main__":
    main()
