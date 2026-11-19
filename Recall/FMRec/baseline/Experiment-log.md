# FMRec baseline — Experiment Log

`Recall/FMRec/baseline/` 的實驗日誌，最新的日期在最上面。超出單一 run 的結論整理到家族的 [INSIGHT.md](../INSIGHT.md)；跨家族比較在 [Recall INSIGHT](../../INSIGHT.md)。

所有 run 都用 2014 Beauty ASIN 切分（131,413 train／22,363 validation，歷史上限 20），validation 對全 catalog 12,101 件排名、不過濾歷史，以 NDCG@10 選最佳。

## 2026-11-19

### 作者預設完整訓練 `fmrec-beauty-v1`（H2）

#### Motivation

FMRec 不用 SID，可作為 TIGER 的另一個 item-ID 對照。檢驗 [FMRec INSIGHT](../INSIGHT.md) H2：同切分下 FMRec 的 validation Recall@10 高於 TIGER baseline。

#### Details

- **Baseline:** 同切分 SASRec `sasrec-beauty-v1`（0.0938）、TIGER `tiger-paper-v1`（0.0368）、熱門商品（0.0170），都是 validation Recall@10。
- **Change:** —（作者程式 `1c39b17` 的預設值；與作者程式的差異見 [baseline README](README.md)）
- **Data:** 2014 Beauty ASIN 切分。
- **Environment:** SoC job 1165520，節點 xgpf11，程式 `bb1088c`，28,388,038 參數，seed 為 `train.py` 預設。每 epoch 約 257 steps，最多 500 epochs，每 20 epochs 驗證一次。
- **Command:** `MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/FMRec/baseline/runs/fmrec-beauty-v1/`；log `slurm-kuriyo-fmrec-1165520.out`（repo 根目錄）。

#### Expectation

validation Recall@10 高於 TIGER 的 0.0342／0.0368；作為成熟的 item-ID 模型，應與 SASRec 同一量級。

#### Key Metrics

- validation Recall@10、NDCG@10 與 best epoch。
- 高於 0.0368 支持 H2。

#### Results

| Run | Recall@10 | NDCG@10 | Notes |
| --- | --- | --- | --- |
| SASRec | 0.0938 | 0.0461 | `sasrec-beauty-v1`，本機 |
| FMRec | 待定 | 待定 | `fmrec-beauty-v1`，job 1165520（訓練中） |

- **Observations:** 截至 log 的 epoch 15（step 3,800），loss 約 2.18 且有限；第一次 validation 在 epoch 20。
- **Hypotheses:** 尚無。

#### Conclusion

進行中。

## 2026-11-18

### CPU smoke `fmrec-smoke-cpu-20261118`

#### Motivation

確認 baseline 的訓練、評估與 checkpoint 流程能在小資料上跑通，再送 GPU 完整訓練。

#### Details

- **Baseline:** 無。
- **Change:** 只取 4,096 筆 train、1,024 筆 validation，2 epochs。
- **Data:** 2014 Beauty ASIN 切分的子集。
- **Environment:** 本機 CPU，約 8.5 分鐘（與完整測試同時跑）。
- **Command:** `python -m Recall.FMRec.baseline.train --run-name fmrec-smoke-cpu-20261118 --epochs 2 --eval-every 1 --max-train-examples 4096 --max-validation-examples 1024`
- **Artifacts:** 本機 `Recall/FMRec/baseline/runs/fmrec-smoke-cpu-20261118/`（Git 忽略）。

#### Expectation

loss 有限、流程跑完；不預期有意義的指標。

#### Key Metrics

- loss 是否有限、評估是否完成。

#### Results

- **Observations:** loss 有限，指標接近 0。
- **Hypotheses:** 無。

#### Conclusion

流程可執行，可以送 GPU 完整訓練；此數字不代表推薦品質。
