---
title: TIGER baseline 訓練與 Adafactor 步長
type: subsystem
created: 2026-11-19
updated: 2026-11-19
sources:
  - code: Recall/TIGER/baseline/train.py
  - code: Recall/TIGER/baseline/sequence.py
  - code: Recall/TIGER/INSIGHT.md
  - code: scripts/run.sbatch
  - code: tests/test_run_sbatch.py
  - code: core/runs.py
  - ext: https://raw.githubusercontent.com/google-research/t5x/main/t5x/adafactor.py
verified_at: 6b9097b
---

# TIGER baseline 訓練與 Adafactor 步長

`Recall/TIGER/baseline/train.py` 是 TIGER 的正式入口。它用 Hugging Face T5，設定依論文：4+4 層、6 heads、d_kv 64、d_ff 1024、dropout 0.1（`Recall/TIGER/baseline/train.py:46`），並加上 2,000 個 hashed user token。每個商品編成 3 個 SID token 加 1 個碰撞 token，各層使用不同的 token offset（`Recall/TIGER/baseline/sequence.py:35`）。學習率前 10k steps 固定 0.01，之後按 inverse square root 衰減。評估用 unconstrained beam search（beam 10）；產生不存在的 SID 時算 miss。

同切分的 SASRec 與 FMRec 達到 validation Recall@10 約 0.10，TIGER 兩個 `scale_parameter=False` 的完整 run 只有 0.034–0.037。改用 T5X 式 Adafactor（SoC job 1165519）後，20k 就達 0.080；證據見 `Recall/TIGER/INSIGHT.md` 的 H5。2026-11-19 起 T5X 式縮放是預設值。

> ⚠️ Hugging Face `Adafactor` 的 `scale_parameter=False` 會把 lr 當成**絕對**步長：更新先以 RMS 正規化，所以每個權重每步大約移動 lr。T5X（論文框架）的 `multiply_by_parameter_scale` 預設為 True，會把 lr 乘上參數 RMS（下限 1e-3）。論文的「lr 0.01」是 T5X 的相對步長；直接搬到 HF 並設 `scale_parameter=False`，HF T5 初始化下 RMS 只有約 0.011 的 `q` 權重，每步會被改動約 90%。T5X 式縮放現在是預設（`train.py` 的 `--adafactor-scale-parameter`、`run.sbatch` 的 `ADAFACTOR_SCALE_PARAMETER` 預設 1）。`--no-adafactor-scale-parameter`／`ADAFACTOR_SCALE_PARAMETER=0` 還原舊行為，只用來續訓 2026-11-19 以前的 run 或重現舊結果。

> ⚠️ 續訓時 Adafactor 的縮放設定必須和原 run 相同。舊 run 的 `config.json` 沒有 `adafactor_scale_parameter` 欄位，視為 False（`Recall/TIGER/baseline/train.py:37`）；預設改成 True 之後，續訓這些舊 run 必須明確加 `ADAFACTOR_SCALE_PARAMETER=0`，否則會被拒絕；要換設定就用新的 `RUN_NAME`。`STAGE=test`（`--eval-test-only`）不檢查這個設定。

相關：[[model-family-layout]]（TIGER baseline 的目錄與 run 位置）、[[run-directory]]（run 版面與舊 run 的搬移）、[[sid-rqvae-training]]（SID 的產生）、[[sasrec]]（同切分的檢查基準）。
