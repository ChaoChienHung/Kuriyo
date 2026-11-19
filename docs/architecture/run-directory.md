---
title: Run 目錄與共用訓練零件
type: architecture
created: 2026-11-19
updated: 2026-11-19
sources:
  - code: core/runs.py
  - code: core/run_logging.py
  - code: core/experiment.py
  - code: core/README.md
  - code: Recall/core/data.py
  - code: Recall/core/evaluation.py
  - code: Recall/SASRec/baseline/train.py
  - code: Recall/FMRec/baseline/train.py
  - code: Recall/TIGER/baseline/train.py
  - code: scripts/run.sbatch
  - code: tests/test_runs.py
verified_at: 6b9097b
---

# Run 目錄與共用訓練零件

SASRec、FMRec 與 TIGER baseline 各自只寫模型與訓練迴圈；run 目錄、設定紀錄、續訓與 test 的檢查、checkpoint、選最佳與 early stopping 都來自 `core/runs.py`。SASRec 與 FMRec 另外共用 `Recall/core/data.py` 的 `AsinSplit`（`Recall/core/data.py:66`）與 `Recall/core/evaluation.py` 的全商品排名（`Recall/core/evaluation.py:11`）。版面與 `settings.json` 的欄位在 `core/README.md`；本頁記為什麼這樣分，以及改動時的坑。

`runs/<RUN_NAME>/` 依用途分成 `settings.json`、`events.log`、`train/`、`checkpoints/`、`validation/<set>/`、`test/<set>/`。`<set>` 是評估集名稱（現在只有 `amazon2014-beauty`），之後加冷啟動等其他協定時各開一個資料夾，結果不會混進同一份 best。`settings.json` 一次記下資料與前處理（split manifest 的規則、來源與筆數）、評估集、模型、訓練、seed 與程式版本，Experiment-log 的 Settings 從這裡抄。

續訓與 test 只比對各模型 `RESUME_KEYS`、`TEST_KEYS` 列出的欄位（例如 `Recall/SASRec/baseline/train.py:31`），用 `check_settings` 逐項比較（`core/runs.py:106`）。字典的鍵是 2026-11-19 以前平的 `config.json` 的欄位名，值是 `settings.json` 裡的路徑；同一張表也用來核對舊 run。

> ⚠️ 共用零件一改，三個模型的結果都可能跟著變。改 `core/runs.py`、`Recall/core/` 之前，先用固定 seed 跑每個受影響模型的小 run（含一次續訓與 test），改完再跑一次，確認 loss、validation 與 test 指標逐位相同。2026-11-19 的重構就是這樣驗證的。

> ⚠️ 舊的平版面 run（`config.json`、`latest.pt`、`validation-step-20000.json`……）在 `--resume` 或 `--eval-test-only` 時由 `migrate_legacy`（`core/runs.py:120`）先核對、再搬移。核對不符就整個拒絕，不會搬一半；相符時會移動檔案、寫 `settings.json`，並保留 `config.json`。不要在舊 run 仍有 job 寫入時執行，否則那個 job 會繼續寫到舊位置。`scripts/run.sbatch` 判斷要不要續訓時，兩種位置的 `latest.pt` 都接受。

> ⚠️ 舊 TIGER run 的 `config.json` 沒有 `adafactor_scale_parameter`，`RESUME_KEYS` 以 `("training.adafactor_scale_parameter", False)` 給預設值（`Recall/TIGER/baseline/train.py:37`）。新增會影響結果的設定時，也要給舊 run 一個符合當時行為的預設值，否則舊 run 會被拒絕續訓。

> ⚠️ `save_checkpoint`（`core/runs.py:202`）把全域 torch 與 CUDA 的 RNG 狀態一起存，`load_checkpoint` 一併還原；`Selection` 的狀態（`best_ndcg10`、`bad_evals`、`stopped_early`）也放在同一個 checkpoint。續訓與不中斷訓練的 loss 相同，靠的是這兩件事，各模型的 `test_train_resume_and_held_out_test` 會檢查。

> ⚠️ `rank_full_catalog` 用 `index_offset` 把分數欄對回 ASIN：SASRec 的第 0 欄就是第一件商品（offset 0），FMRec 的第 0 欄是 padding（offset 1）。給錯 offset 不會報錯，只會讓所有命中錯開一件。

TIGER 的 smoke trainer（`Recall/TIGER/baseline/train_sequence.py`）仍用舊的平版面；SID 產生器只用 `core/run_logging.py`，不經過 `core/runs.py`。

相關：[[model-family-layout]]（家族與變體的目錄規則）、[[tiger-training]]、[[sasrec]]、[[fmrec]]。
