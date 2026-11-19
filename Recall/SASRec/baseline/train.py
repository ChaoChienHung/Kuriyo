"""Train and evaluate the SASRec baseline on the prepared Amazon 2014 ASIN split.

A sanity baseline: full-catalog ranking on the same leave-one-out split and the
same shared metrics as TIGER and FMRec. Validation selects the checkpoint;
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

from Recall.SASRec.baseline.data import collate_histories, collate_windows, user_sequences, windows
from Recall.SASRec.baseline.model import LOSSES, SASRec
from Recall.core.data import AsinSplit
from Recall.core.evaluation import rank_full_catalog
from core.experiment import seed_everything
from core.runs import (RunError, RunLayout, Selection, add_run_arguments, check_run_arguments,
                       configure_threads, evaluate_test_once, load_checkpoint, open_run,
                       run_section, save_checkpoint)

MODEL_KEYS = ("max_history", "hidden", "heads", "layers", "dropout")
# Must match to resume; keys are the flat config.json names of runs before 2026-11-19.
RESUME_KEYS = {"train_sha256": "data.splits.train.sha256",
               "validation_sha256": "data.splits.validation.sha256",
               "catalog_sha256": "data.catalog.sha256", "seed": "run.seed",
               "batch_size": "training.batch_size",
               "max_train_examples": "data.max_train_examples", "model": "model.config",
               "loss": "training.loss", "lr": "training.lr",
               "weight_decay": "training.weight_decay"}
TEST_KEYS = {key: RESUME_KEYS[key] for key in
             ("catalog_sha256", "train_sha256", "validation_sha256", "model")}


def evaluate(model, rows, asins, max_history, device, batch_size, ks=(5, 10)):
    """Full ranking over every catalog item from the newest position."""
    model.eval()

    def score(batch):
        sequence, _ = collate_histories(batch, max_history)
        return model.scores(model(sequence.to(device))[:, -1])

    with torch.no_grad():
        return rank_full_catalog(rows, asins, score, batch_size, model.num_items, ks)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/amazon_reviews")
    parser.add_argument("--category", default="Beauty")
    add_run_arguments(parser, ROOT / "Recall/SASRec/baseline/runs")
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    error = check_run_arguments(args)
    if error:
        parser.error(error)
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_every,
           args.max_history, args.hidden, args.heads, args.layers) < 1 or args.patience < 0:
        parser.error("counts must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    try:
        split = AsinSplit(args.data_root, args.category)
        split.check_history(args.max_history)
    except RunError as error:
        parser.error(str(error))
    asins = split.asins

    model_config = {key: getattr(args, key) for key in MODEL_KEYS}
    layout = RunLayout(args.runs_dir / args.run_name, split.eval_set)
    settings = {
        "run": run_section(args.run_name, "SASRec", "baseline",
                           "Kuriyo PyTorch SASRec (Kang & McAuley 2018)", ROOT, args.device, args.seed),
        "data": {**split.settings(ROOT, args.max_history, args.max_train_examples,
                                  args.max_validation_examples),
                 "training_examples": "per-user train sequences rebuilt from prefix examples, split "
                                      "into windows of max_history + 1; loss at every real position"},
        "model": {"config": model_config},
        "training": {"loss": args.loss, "optimizer": "Adam(betas=(0.9, 0.98))", "lr": args.lr,
                     "weight_decay": args.weight_decay, "epochs": args.epochs,
                     "batch_size": args.batch_size},
        "evaluation": {"set": split.eval_set, "every_epochs": args.eval_every,
                       "batch_size": args.eval_batch_size, "patience": args.patience,
                       "selection": "best validation NDCG@10", "ks": [5, 10],
                       "ranking": "full catalog, inner product with item embeddings; "
                                  "history items not filtered"}}

    seed_everything(args.seed)
    configure_threads(args.device)
    model = SASRec(len(asins), **model_config).to(args.device)
    settings["model"]["parameters"] = sum(parameter.numel() for parameter in model.parameters())

    if args.eval_test_only:
        test = split.read("test")
        try:
            evaluate_test_once(layout, settings, TEST_KEYS, RESUME_KEYS, model,
                               lambda: evaluate(model, test, asins, args.max_history, args.device,
                                                args.eval_batch_size),
                               split.sha256["test"])
        except RunError as error:
            parser.error(str(error))
        return

    train_rows = split.read("train", args.max_train_examples)
    train = [window for sequence in user_sequences(train_rows, args.max_history)
             for window in windows(sequence, args.max_history)]
    validation = split.read("validation", args.max_validation_examples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                                 weight_decay=args.weight_decay)
    try:
        recorded = open_run(layout, settings, RESUME_KEYS, args.resume)
    except RunError as error:
        parser.error(str(error))
    if args.resume:
        state = load_checkpoint(layout.latest, args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
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
        print("SASRec parameters={} items={} train_examples={} windows={} validation={} "
              "starting_epoch={}".format(settings["model"]["parameters"], len(asins), len(train_rows),
                                         len(train), len(validation), start_epoch), flush=True)
        if selection.stopped:
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
                metrics.update(split="validation", set=layout.eval_set, epoch=epoch, step=step)
                selection.record(layout, logger, model, metrics, "epoch", epoch)
            save_checkpoint(layout.latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
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
