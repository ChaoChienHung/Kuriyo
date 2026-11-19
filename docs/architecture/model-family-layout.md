---
title: 模型家族與變體目錄結構
type: architecture
created: 2026-11-17
updated: 2026-11-19
sources:
  - code: Recall/TIGER/README.md
  - code: Recall/TIGER/baseline/train.py
  - code: Recall/TIGER/baseline/train_sequence.py
  - code: Recall/TIGER/baseline/sequence.py
  - code: Recall/TIGER/baseline/requirements.txt
  - code: Recall/TIGER/baseline/tests/test_train.py
  - code: scripts/run.sbatch
  - code: scripts/train_tiger_sequence.py
  - code: tests/test_run_sbatch.py
  - code: AGENTS.md
  - code: Recall/TIGER/Experiment-log.md
  - code: Recall/FMRec/README.md
  - code: Recall/SASRec/README.md
  - code: .gitignore
verified_at: 4f6594d
---

# 模型家族與變體目錄結構

Kuriyo 以「模型家族」組織同一類模型的不同實作，**每個變體自成一個完整的空間**。目前有三個家族：`Recall/TIGER/`、`Recall/FMRec/`（見 [[fmrec]]）與 `Recall/SASRec/`（見 [[sasrec]]），各自只有 `baseline/` 變體。規範原文在 `AGENTS.md` 的 "Model families, variants, and tests" 一節；本頁說明實際怎麼運作，以及改動時容易踩到的地方。

## 結構

- **家族根層**只放 `README.md`（家族說明與變體表）、`INSIGHT.md`（適用於所有變體的研究筆記）、`Experiment-log.md`（所有變體的實驗日誌，每個條目標明 Variant）和套件用的初始化檔，不放程式、測試、依賴或 runs。研究筆記放在家族層，因為像「SID 品質影響效果」這類發現對每個變體都成立；實驗日誌也放在家族層，才能在同一份檔案比較不同變體的 run。
- **每個變體**是家族下的直接子目錄，擁有自己的 `README.md`、`requirements.txt`、模型與訓練程式、`tests/` 與 `runs/<RUN_NAME>/`（版面見 [[run-directory]]）。run 目錄、checkpoint 與評估流程等多個家族共用的結構放在 `core/` 與 `Recall/core/`，變體只寫模型與訓練迴圈。baseline 的內容：
  - 正式入口 `python -m Recall.TIGER.baseline.train`，預設輸出目錄在 `Recall/TIGER/baseline/train.py:112`。
  - `SIDTokenizer`（`Recall/TIGER/baseline/sequence.py:10`）與 constrained beam search（`Recall/TIGER/baseline/sequence.py:67`）。
  - smoke trainer `Recall/TIGER/baseline/train_sequence.py`，其 runs 也寫到 baseline 自己的目錄（`Recall/TIGER/baseline/train_sequence.py:24`）。根目錄的相容 wrapper `scripts/train_tiger_sequence.py` 轉交給它。
  - 依賴 `Recall/TIGER/baseline/requirements.txt`，以 `-r ../../../requirements.txt` 引入根目錄共用依賴；`scripts/run.sbatch` 安裝的就是這份。
- **變體之間不互相 import**，只可 import `Recall/core/`、`core/`、`Representations/` 等跨模型共用元件。這樣修改 baseline 不會悄悄改變其他變體的結果；代價是新變體通常要從複製 `baseline/` 開始。
- `.gitignore` 對 `Recall/**/runs/*/` 用白名單：只追蹤 `settings.json`、`events.log`、`validation/`、`test/`，checkpoint、`train/` 與快取都忽略。
- **變體有 run 記進 Experiment-log 後結果凍結**（`AGENTS.md`）：同樣設定之後要得到同樣結果；改預設值、換演算法或修正改變結果的 bug 要開新變體。凍結前的例外列在 SASRec 與 TIGER baseline 的 README。

`scripts/run.sbatch` 以 `MODEL`（tiger／sasrec／fmrec，預設 tiger）選擇家族，每個家族目前都跑 `baseline/` 變體的 `train.py`，runs 寫在 `Recall/<Family>/baseline/runs/<RUN_NAME>/`（`scripts/run.sbatch:98` 起）。SASRec 與 FMRec 不用 SID，只接受 `STAGE=prepare|train|test|all`；`CATEGORY`（預設 `Beauty`）選資料集，`RUN_NAME` 預設為 `sasrec-paper-<分類小寫>-v1`、`fmrec-<分類小寫>-v1`（既有的 `sasrec-beauty-v1` 是舊預設的 CE 設定），TIGER 在 Beauty 以外是 `tiger-paper-<分類小寫>-v1`。`SEED` 與 `TRAIN_ARGS` 會同時傳給 train 與 test。

> ⚠️ 同時提交多個 job（不同 `MODEL`）時用 `STAGE=train`。`STAGE=all`／`prepare` 會讓每個 job 都執行 `scripts.ensure_amazon_reviews_2014`，同時寫同一份分類切分，沒有鎖保護。`STAGE=train` 只檢查切分存在（`tests/test_run_sbatch.py` 有對應測試）。

只有超參數不同時，在既有變體加具名設定；確實需要不同實作時，才新增與 `baseline/` 同層的目錄，名稱描述差異（例如 `constrained-decoding/`），不用 `variant1/`、`v2/`，並加入家族 README 的變體表。

## 歷史

2026-11-17 起，原本在 `Recall/TIGER/` 下的全部程式（paper trainer、`sequence.py`、smoke trainer、`config.py`、`requirements.txt`、tests）都搬進 `Recall/TIGER/baseline/`，邏輯未改。run 的預設位置也從 `Recall/TIGER/runs/` 改到 `Recall/TIGER/baseline/runs/`。SoC job 1153307 的 run `tiger-paper-20261117-005232` 是在舊路徑產生的。

> ⚠️ 變體的程式在 repo 根目錄下第三層，所以 `Recall/TIGER/baseline/train.py:18` 用 `parents[3]` 取得 repo 根目錄；但變體的 `tests/` 多一層，要用 `parents[4]`（`Recall/TIGER/baseline/tests/test_train.py:13`）。同理，變體 `requirements.txt` 引用根目錄依賴要用 `../../../`。搬移或複製變體時若沒有一起調整，subprocess 會在錯誤的 cwd 下執行，pip 也會找不到根目錄 requirements。

> ⚠️ 改變某個變體的 runs 位置時，`scripts/run.sbatch` 必須拒絕「舊路徑有 run、新路徑沒有」的情況（`scripts/run.sbatch:187` 起），否則 `STAGE=train` 會在新路徑從頭開一個同名 run，而不是續訓。對應測試在 `tests/test_run_sbatch.py`。SoC 上的舊 run 需手動執行 `mv Recall/TIGER/runs/<RUN_NAME> Recall/TIGER/baseline/runs/`。

> ⚠️ 複製 `baseline/` 建立新變體時，要把新目錄裡所有 `Recall.TIGER.baseline.` 的 import 改成新變體自己的模組名稱。漏改的話新變體仍會執行 baseline 的程式，違反「變體不互相 import」，而且測試照樣會通過，不容易發現。

> ⚠️ 每個變體目錄與其 `tests/` 都要有套件初始化檔，否則根目錄的 `python -m unittest discover -s .` 會悄悄跳過該變體的測試。

相關：TIGER 的完整操作流程見 `Recall/TIGER/baseline/README.md`。
