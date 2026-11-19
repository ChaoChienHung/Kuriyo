# SASRec — Experiment Log

`Recall/SASRec/` 家族所有變體的實驗日誌，最新的日期在最上面；每個條目的 Details 以 **Variant** 標明變體，放在一起方便跨變體比較。超出單一 run 的結論整理到家族的 [INSIGHT.md](INSIGHT.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

所有 run 都用 2014 Beauty ASIN 切分（131,413 train／22,363 validation，歷史上限 20），validation 對全 catalog 12,101 件排名、不過濾歷史，以 NDCG@10 選最佳。

## 2026-11-19

### SoC GPU 重跑 `sasrec-beauty-v1`

#### Motivation

和 TIGER、FMRec 一起在 SoC 跑，確認 `run.sbatch` 的 `MODEL=sasrec` 能在 GPU 上完成，並得到一份有 SoC log 與產物的 run。

#### Details

- **Variant:** `baseline`
- **Baseline:** 本機 run `sasrec-beauty-v1`（同日下方條目）。
- **Change:** 只換執行環境（本機 MPS → SoC CUDA）；設定與 seed 相同。
- **Data:** 2014 Beauty ASIN 切分。
- **Environment:** SoC job 1165570，程式 `7c00c35`。先前的 job 1165521（xgpe4）與 1165548（xgpe6，程式 `6c4d233`）都因 `CUDA unknown error` 在建立 run 目錄前以 exit 2 結束，沒有訓練。
- **Command:** `MODEL=sasrec RUN_NAME=sasrec-beauty-v1 STAGE=train sbatch --export=ALL --exclude='xgpe[0-4,6]' --job-name=kuriyo-sasrec scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/SASRec/baseline/runs/sasrec-beauty-v1/`；log `slurm-kuriyo-sasrec-1165570.out`（repo 根目錄）。失敗 log `slurm-kuriyo-sasrec-1165548.out`，1165521 用 `git show b7a4078:slurm-kuriyo-sasrec-1165521.out` 取回。

#### Settings

- **資料／前處理：**Amazon Reviews 2014 Beauty 5-core（22,363 users、12,101 items）；每位使用者依 `unixReviewTime` 排序、同時間保留原始列序，最後一筆是 test、倒數第二筆是 validation，其餘每個前綴一筆 train 樣本（131,413 筆）；歷史截到最近 20 件，重複互動保留（`scripts.prepare_amazon_reviews_2014`）。商品 ID embedding 隨機初始化，不使用商品文字。
- **模型：**hidden 64、2 層、2 heads、dropout 0.5，可學的位置 embedding，歷史上限 20；826,368 參數。
- **訓練：**full-softmax CE，Adam（betas 0.9、0.98）lr 1e-3、weight decay 0，batch 128 個視窗，最多 200 epochs，seed 42。
- **評估：**每 5 epochs 做一次 validation，以 NDCG@10 選 `best.pt`，patience 5。

#### Expectation

GPU 與 MPS 的數值不會逐位相同，但 best validation Recall@10 應落在本機的 0.092–0.094 附近。

#### Key Metrics

- best validation Recall@10、NDCG@10 與 best epoch。
- 與本機相差超過約 0.005 就要查環境差異。

#### Results

| Run | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Notes |
| --- | --- | --- | --- | --- | --- |
| 本機 MPS（best epoch 50） | 0.06394 | 0.09382 | 0.03647 | 0.04609 | 早停於 epoch 75 |
| SoC CUDA（best epoch 85） | 0.06882 | 0.10227 | 0.04010 | 0.05083 | job 1165570，早停於 epoch 110 |

| epoch | 5 | 10 | 20 | 30 | 50 | 85 | 110 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SoC Recall@10 | 0.03421 | 0.06332 | 0.09230 | 0.09842 | 0.10236 | 0.10227 | 0.10231 |

- **Observations:** 前兩次提交落在壞節點；排除整組 xgpe 後的 job 1165570 跑完並存檔。SoC 的 best Recall@10 比本機高 0.0085、NDCG@10 高 0.0047，超出預期的範圍。SoC 在 epoch 35 之後停在 0.101–0.103，比本機的平台（0.092–0.094）高，訓練也多走了 35 個 epoch 才早停。
- **Hypotheses:** 差異可能來自 MPS 與 CUDA 的數值與隨機數不同；兩邊各只有一個 seed，無法分辨環境差異和 seed 變異。未驗證。 冷啟動限制：商品 embedding 隨機初始化、只從互動學習，沒在 train 出現過或很少出現的商品學不好；這個平均指標由熱門商品主導，看不出這點。本 run 沒有依目標商品的 train 出現次數分組評估（`Recall/INSIGHT.md` H2）。

#### Conclusion

SoC run 完成，作為 SASRec 的主要參考（有完整 commit 紀錄，本機 run 沒有）。與本機差 0.0085，原因不明；它不影響「共用資料與評估正常」的結論，因為兩者都遠高於 TIGER。若要釐清，需在同一環境多跑幾個 seed。沒有跑 test。

### 本機完整 run `sasrec-beauty-v1`

#### Motivation

TIGER 換了健康的新 SID 後 validation 仍只有 0.0368。用成熟的 item-ID 模型在同一切分、同一指標下跑一次，判斷問題在共用的資料與評估，還是在 TIGER 自己的流程。

#### Details

- **Variant:** `baseline`
- **Baseline:** TIGER 論文表 3 的 SASRec（Recall@10 0.0605，test，不同前處理）；同切分 TIGER `tiger-paper-v1`（0.0368）。
- **Change:** —
- **Data:** 2014 Beauty ASIN 切分；131,413 個 train 前綴還原成 23,607 個視窗。
- **Environment:** 本機 Apple MPS，約 23 分鐘。
- **Command:** `python -m Recall.SASRec.baseline.train --run-name sasrec-beauty-v1 --device mps`
- **Artifacts:** 本機 `Recall/SASRec/baseline/runs/sasrec-beauty-v1/`（Git 忽略），`best.pt` 是 epoch 50。`config.json` 記錄的 revision 是 `97cae0f`，但 SASRec 程式當時尚未 commit，內容與之後提交的版本相同。

#### Settings

- **資料／前處理：**Amazon Reviews 2014 Beauty 5-core（22,363 users、12,101 items）；每位使用者依 `unixReviewTime` 排序、同時間保留原始列序，最後一筆是 test、倒數第二筆是 validation，其餘每個前綴一筆 train 樣本（131,413 筆）；歷史截到最近 20 件，重複互動保留（`scripts.prepare_amazon_reviews_2014`）。商品 ID embedding 隨機初始化，不使用商品文字。
- **模型：**hidden 64、2 層、2 heads、dropout 0.5，可學的位置 embedding，歷史上限 20；826,368 參數。
- **訓練：**full-softmax CE，Adam（betas 0.9、0.98）lr 1e-3、weight decay 0，batch 128 個視窗，最多 200 epochs，seed 42。
- **評估：**每 5 epochs 做一次 validation，以 NDCG@10 選 `best.pt`，patience 5。

#### Expectation

若共用資料與評估正確，Recall@10 應接近論文的 0.0605；若也只有 0.03 左右，問題在共用部分。

#### Key Metrics

- best validation Recall@10、NDCG@10。

#### Results

| epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| 5 | 0.02039 | 0.03349 | 0.01172 | 0.01593 |
| 10 | 0.04047 | 0.06122 | 0.02344 | 0.03012 |
| 20 | 0.05862 | 0.08612 | 0.03323 | 0.04211 |
| 30 | 0.06310 | 0.09332 | 0.03600 | 0.04572 |
| **50（best NDCG@10）** | **0.06394** | **0.09382** | **0.03647** | **0.04609** |
| 60 | 0.06453 | 0.09422 | 0.03653 | 0.04604 |
| 75（early stop） | 0.06202 | 0.09189 | 0.03610 | 0.04573 |

- **Observations:** epoch 30 之後在 Recall@10 0.092–0.094 之間波動，epoch 75 early stop。結果比論文的 0.0605 還高，約為同切分 TIGER 的 3 倍。
- **Hypotheses:** 高於論文主要來自 full-softmax CE。論文引用的 SASRec 多為 BCE，每個正例只配一個隨機負例；隨機負例大多很容易分開，學到的界線只區分目標與無關商品。CE 每步都和全部商品比較，分數高的難負例梯度最大，直接對應全商品排名。未驗證，記為 [SASRec INSIGHT](INSIGHT.md) H1（`--loss bce` 對照）。

#### Conclusion

支持「共用資料與評估正常」：TIGER 的差距在它自己的流程。已整理到 [Recall INSIGHT](../INSIGHT.md) Insights。模型未定案，沒有跑 test。
