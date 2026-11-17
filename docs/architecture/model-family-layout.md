---
title: 模型家族與變體目錄結構
type: architecture
created: 2026-11-17
updated: 2026-11-17
sources:
  - code: Recall/TIGER/baseline/train.py
  - code: Recall/TIGER/baseline/tests/test_train.py
  - code: Recall/TIGER/scripts/train_sequence.py
  - code: Recall/TIGER/sequence.py
  - code: scripts/run.sbatch
  - code: tests/test_run_sbatch.py
  - code: AGENTS.md
  - code: .gitignore
verified_at: 6d1d22b
---

# 模型家族與變體目錄結構

Kuriyo 以「模型家族」組織同一類模型的不同實作。目前唯一的家族是 `Recall/TIGER/`，唯一的變體是 `baseline/`。規範原文在 `AGENTS.md` 的 "Model families, variants, and tests" 一節；本頁說明這套結構實際怎麼運作，以及改動時容易踩到的地方。

## 結構

- **家族根層**放各變體共用的程式：`Recall/TIGER/sequence.py:10` 的 `SIDTokenizer`、`Recall/TIGER/sequence.py:67` 的 constrained beam search，以及 smoke trainer `Recall/TIGER/scripts/train_sequence.py`（其 `read_examples` 在 `Recall/TIGER/scripts/train_sequence.py:27`，baseline 也會 import）。家族共用測試放 `Recall/TIGER/tests/`。
- **每個變體**是家族下的直接子目錄，與其他變體同層，包含 `README.md`、`__init__.py`、入口程式、`tests/` 與 Git 忽略的 `runs/<RUN_NAME>/`。baseline 的入口為 `python -m Recall.TIGER.baseline.train`，預設輸出目錄在 `Recall/TIGER/baseline/train.py:113`。
- **家族層 `runs/`** 只放 smoke trainer 的產物（`Recall/TIGER/scripts/train_sequence.py:24`）。
- `.gitignore` 以 `Recall/**/runs/` 涵蓋所有家族與變體的 runs。

只有超參數不同時，在既有變體加具名設定；確實需要不同實作時，才新增與 `baseline/` 同層的目錄，名稱描述差異（例如 `constrained-decoding/`），不用 `variant1/`、`v2/`。新增後要加入 `Recall/TIGER/README.md` 的變體表。

## 歷史

2026-11-17 起，baseline 由 `Recall/TIGER/scripts/train_paper_baseline.py` 搬到 `Recall/TIGER/baseline/train.py`，程式本身未改。run 的預設位置也從 `Recall/TIGER/runs/` 改到 `Recall/TIGER/baseline/runs/`。SoC job 1153307 的 run `tiger-paper-20261117-005232` 是在舊路徑產生的。

> ⚠️ 變體目錄和 `scripts/` 一樣在 repo 根目錄下第三層，所以 `Recall/TIGER/baseline/train.py:18` 用 `parents[3]` 取得 repo 根目錄；但變體的 `tests/` 多一層，要用 `parents[4]`（`Recall/TIGER/baseline/tests/test_train.py:13`）。搬移檔案時若沒有一起調整，subprocess 會在錯誤的 cwd 下執行，找不到 `Recall` 套件。

> ⚠️ 改變某個變體的 runs 位置時，`scripts/run.sbatch` 必須拒絕「舊路徑有 run、新路徑沒有」的情況（`scripts/run.sbatch:91`），否則 `STAGE=train` 會在新路徑從頭開一個同名 run，而不是續訓。對應測試在 `tests/test_run_sbatch.py`。SoC 上的舊 run 需手動執行 `mv Recall/TIGER/runs/<RUN_NAME> Recall/TIGER/baseline/runs/`。

> ⚠️ 每個變體目錄與其 `tests/` 都要有 `__init__.py`，否則根目錄的 `python -m unittest discover -s .` 會悄悄跳過該變體的測試。

相關：TIGER 的操作流程見 `Recall/TIGER/README.md`；baseline 細節見 `Recall/TIGER/baseline/README.md`。
