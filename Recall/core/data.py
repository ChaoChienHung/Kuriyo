"""Data helpers shared by Recall models: prepared ASIN splits and token batches."""

import json
from pathlib import Path

import torch

from core.experiment import file_sha256
from core.runs import RunError, display_path, eval_set_name

SPLITS = ("train", "validation", "test")


def pad_token_sequences(sequences, pad_id=0):
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("sequences must be nonempty")
    padded = torch.full((len(sequences), max(map(len, sequences))), pad_id,
                        dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded[index, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
    return padded


def load_catalog(path):
    """Return ASINs in catalog order; item index i + 1 belongs to asins[i]."""
    with path.open(encoding="utf-8") as source:
        asins = [json.loads(line)["asin"] for line in source if line.strip()]
    if len(set(asins)) != len(asins):
        raise ValueError("duplicate ASIN in catalog {}".format(path))
    return asins


def read_examples(path, item_index, limit=None):
    """Rows with integer history/target; unknown ASINs are an error, not a skip."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            try:
                history = [item_index[asin] for asin in row["history"]]
                target = item_index[row["target_asin"]]
            except KeyError as error:
                raise ValueError("{} has an ASIN outside the catalog: {}".format(path, error))
            if not history:
                raise ValueError("{} has an empty history".format(path))
            rows.append({"user_id": row["user_id"], "history": history, "target": target})
    if not rows:
        raise ValueError("no examples in {}".format(path))
    return rows


def preprocessing_summary(manifest_path, root):
    """What scripts.prepare_amazon_reviews_2014 recorded about how the split was built."""
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    return {"producer": "scripts.prepare_amazon_reviews_2014",
            "manifest": {"path": display_path(manifest_path, root),
                         "sha256": file_sha256(manifest_path)},
            "dataset": manifest.get("dataset"), "source": manifest.get("source"),
            "rules": manifest.get("rules"), "counts": manifest.get("counts")}


class AsinSplit:
    """The prepared per-user leave-one-out ASIN split of one Amazon 2014 category."""

    def __init__(self, data_root, category, year="2014"):
        self.year, self.category = year, category
        base = Path(data_root) / year
        self.catalog_path = base / "catalog" / category / "items.jsonl"
        self.manifest_path = base / "train/raw" / category / "split_manifest.json"
        self.paths = {split: base / split / "raw" / category / "examples.jsonl" for split in SPLITS}
        if not self.catalog_path.is_file() or not self.manifest_path.is_file() or not all(
                path.is_file() for path in self.paths.values()):
            raise RunError("prepare the {} {} split first (scripts.prepare_amazon_reviews_2014)".format(
                year, category))
        self.catalog_sha256 = file_sha256(self.catalog_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("catalog", {}).get("sha256") not in (None, self.catalog_sha256):
            raise RunError("split manifest was produced from a different catalog")
        self.asins = load_catalog(self.catalog_path)
        self.item_index = {asin: index + 1 for index, asin in enumerate(self.asins)}
        self.sha256 = {split: file_sha256(path) for split, path in self.paths.items()}

    @property
    def eval_set(self):
        return eval_set_name(self.year, self.category)

    def check_history(self, max_history):
        cap = self.manifest.get("rules", {}).get("max_history")
        if cap is not None and max_history > cap:
            raise RunError("--max-history exceeds the split's history cap {}".format(cap))

    def read(self, split, limit=None):
        return read_examples(self.paths[split], self.item_index, limit)

    def settings(self, root, max_history, max_train_examples, max_validation_examples):
        return {"year": self.year, "category": self.category,
                "catalog": {"path": display_path(self.catalog_path, root),
                            "sha256": self.catalog_sha256, "items": len(self.asins)},
                "splits": {split: {"path": display_path(path, root), "sha256": self.sha256[split]}
                           for split, path in self.paths.items()},
                "preprocessing": preprocessing_summary(self.manifest_path, root),
                "max_history": max_history, "max_train_examples": max_train_examples,
                "max_validation_examples": max_validation_examples}
