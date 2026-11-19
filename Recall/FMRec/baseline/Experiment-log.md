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
- **Environment:** SoC job 1165520，節點 xgpf11，程式 `bb1088c`，28,388,038 參數。每 epoch 約 257 steps，最多 500 epochs，每 20 epochs 驗證一次。
- **Command:** `MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/FMRec/baseline/runs/fmrec-beauty-v1/`；log `slurm-kuriyo-fmrec-1165520.out`（repo 根目錄）。

#### Settings

- **模型：**hidden 128、4 heads，Decoder1／Decoder2 各 2 層，dropout 0.1、embedding dropout 0.3，attention 雙向（遮 padding key），歷史上限 20；28,388,038 參數。
- **Flow matching：**時間取樣 mode（s=1.0），Euler 30 步，`lambda_uncertainty` 0.001、eps 0.001；L_CE 權重 α 0.2、L_MSE 權重 β 0.4，mask ratio 1.0。
- **訓練：**Adam lr 1e-3、weight decay 0，StepLR 每 100 epochs ×0.1，batch 512，最多 500 epochs，seed 42。
- **評估：**每 20 epochs 做一次 validation，以 NDCG@10 選最佳，patience 5。

#### Expectation

validation Recall@10 高於 TIGER 的 0.0342／0.0368；作為成熟的 item-ID 模型，應與 SASRec 同一量級。

#### Key Metrics

- validation Recall@10、NDCG@10 與 best epoch。
- 高於 0.0368 支持 H2。

#### Results

| Run | Recall@10 | NDCG@10 | Notes |
| --- | --- | --- | --- |
| SASRec | 0.1023 | 0.0508 | `sasrec-beauty-v1`，SoC job 1165570 |
| FMRec epoch 20 | 0.05701 | 0.03300 | job 1165520 |
| FMRec epoch 40 | 0.08751 | 0.05070 | |
| FMRec epoch 60 | 0.09936 | 0.05767 | |
| FMRec epoch 80（目前 best NDCG@10） | 0.10186 | 0.05954 | Recall@5 0.07119、NDCG@5 0.04961 |
| FMRec epoch 100 | 0.10186 | 0.05950 | 訓練中，StepLR 已把 lr 降到 1e-4 |

- **Observations:** 截至 log 的 epoch 108，已做 5 次 validation。epoch 80 的 Recall@10 0.1019 已和 SASRec 相當，NDCG@10 0.0595 比 SASRec 高 0.009；是 TIGER 新 SID 200k（0.0368）的約 2.8 倍。epoch 80 到 100 幾乎沒變。
- **Hypotheses:** 尚無。

#### Conclusion

進行中，但已支持 H2：FMRec 遠高於 TIGER baseline。等早停或 500 epochs 後記錄最終 best。
