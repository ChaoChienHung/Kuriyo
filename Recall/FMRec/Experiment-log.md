# FMRec — Experiment Log

`Recall/FMRec/` 家族所有變體的實驗日誌，最新的日期在最上面；每個條目的 Details 以 **Variant** 標明變體，放在一起方便跨變體比較。超出單一 run 的結論整理到家族的 [INSIGHT.md](INSIGHT.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

所有 run 都用 2014 Beauty ASIN 切分（131,413 train／22,363 validation，歷史上限 20），validation 對全 catalog 12,101 件排名、不過濾歷史，以 NDCG@10 選最佳。

## 2026-11-19

### 作者預設完整訓練 `fmrec-beauty-v1`（H2）

#### Motivation

FMRec 不用 SID，可作為 TIGER 的另一個 item-ID 對照。檢驗 [FMRec INSIGHT](INSIGHT.md) H2：同切分下 FMRec 的 validation Recall@10 高於 TIGER baseline。

#### Details

- **Variant:** `baseline`
- **Baseline:** 同切分 SASRec `sasrec-beauty-v1`（0.0938）、TIGER `tiger-paper-v1`（0.0368）、熱門商品（0.0170），都是 validation Recall@10。
- **Change:** —（作者程式 `1c39b17` 的預設值；與作者程式的差異見 [baseline README](baseline/README.md)）
- **Data:** 2014 Beauty ASIN 切分。
- **Environment:** SoC job 1165520，節點 xgpf11，程式 `bb1088c`，28,388,038 參數。每 epoch 約 257 steps，最多 500 epochs，每 20 epochs 驗證一次。
- **Command:** `MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/FMRec/baseline/runs/fmrec-beauty-v1/`；log `slurm-kuriyo-fmrec-1165520.out`（repo 根目錄）。

#### Settings

- **資料／前處理：**Amazon Reviews 2014 Beauty 5-core（22,363 users、12,101 items）；每位使用者依 `unixReviewTime` 排序、同時間保留原始列序，最後一筆是 test、倒數第二筆是 validation，其餘每個前綴一筆 train 樣本（131,413 筆）；歷史截到最近 20 件，重複互動保留（`scripts.prepare_amazon_reviews_2014`）。商品 ID embedding 隨機初始化，不使用商品文字。
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
| FMRec epoch 80 | 0.10186 | 0.05954 | |
| FMRec epoch 100 | 0.10186 | 0.05950 | StepLR 把 lr 降到 1e-4 |
| FMRec epoch 120 | 0.10325 | 0.06014 | |
| FMRec epoch 140 | 0.10316 | 0.06025 | |
| **FMRec epoch 160（best NDCG@10）** | **0.10307** | **0.06035** | Recall@5 0.07280、NDCG@5 0.05058 |
| FMRec epoch 200 | 0.10218 | 0.06011 | lr 降到 1e-5 |
| FMRec epoch 260（early stop） | 0.10204 | 0.06010 | |

- **Observations:** 在 epoch 260 early stop 並存檔（step 66,800）。best 在 epoch 160：Recall@10 0.1031 與 SASRec（0.1023）相當，NDCG@10 0.0603 比 SASRec 高約 0.010，是 TIGER 新 SID 200k（0.0368）的約 2.8 倍。epoch 80 之後只在 0.102–0.103 之間變動，兩次 lr 衰減幾乎沒有幫助。
- **Hypotheses:** 冷啟動限制：商品 embedding 隨機初始化、只從互動學習，沒在 train 出現過或很少出現的商品學不好；這個平均指標由熱門商品主導，看不出這點。本 run 沒有依目標商品的 train 出現次數分組評估（`Recall/INSIGHT.md` H2）。

#### Conclusion

支持 H2：FMRec 遠高於當時的 TIGER baseline。`best.pt` 是 epoch 160；模型未定案，沒有跑 test。NDCG 高於 SASRec 而 Recall 相當，代表 FMRec 把命中的商品排得更前面。
