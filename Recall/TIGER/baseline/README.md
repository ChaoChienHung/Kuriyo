# TIGER baseline

`baseline/` 是 TIGER 家族目前的比較起點：依 [TIGER 論文](https://arxiv.org/abs/2305.05065)已公開設定建立的 PyTorch／Hugging Face T5 實作，不是作者的 T5X 程式。`baseline` 代表後續變體的比較對象，**不代表已重現論文數值**。

SoC job 1153307 以此實作（當時位於 `Recall/TIGER/scripts/train_paper_baseline.py`，程式 `5f99955`）完成首輪 200,000-step 訓練與 validation，run 為 `tiger-paper-20261117-005232`；尚無最終 test 結果。

## 執行

從 repository 根目錄執行。分階段 batch 流程（`STAGE=train|test`）、依賴安裝與前置資料見 [TIGER README](../README.md)；`scripts/run.sbatch` 呼叫的就是此入口。

```bash
.venv/bin/python -m Recall.TIGER.baseline.train \
  --sid-run "$SID_DIR" --sid-method sentence-t5_rqvae \
  --run-name "$RUN_NAME" --device cuda
```

續訓加 `--resume`；模型定案後的最終 held-out test 加 `--eval-test-only`。輸出寫到 `Recall/TIGER/runs/<RUN_NAME>/`（Git 忽略）；預設設定與產物說明見 [TIGER README](../README.md#家族結構與變體)。

## 與家族共用的部分

- [`../sequence.py`](../sequence.py)：`SIDTokenizer`。
- [`../scripts/train_sequence.py`](../scripts/train_sequence.py)：`read_examples`。
- [`Recall/core/`](../../core/README.md)：padding、metrics、seed 與實驗記錄；[`core/run_logging.py`](../../../core/run_logging.py)：`train.jsonl`／`events.log`。

## 解碼與評估

Unconstrained beam search（beam 10）；無效 SID 與重複商品保留排名位置並計為未命中。validation 依 NDCG@10 選 `best.pt`；test 只在 `--eval-test-only` 時執行，且拒絕覆寫既有 `test-metrics.json`。

## 測試

[`tests/test_train.py`](tests/test_train.py) 以 CPU 小資料驗證 materialize → 1 step 訓練 → 續訓 → held-out test，以及 `train.jsonl`／`events.log` 的 session 與事件紀錄。從根目錄執行 `.venv/bin/python -m unittest discover -s . -v` 會一併跑家族與共用測試。
