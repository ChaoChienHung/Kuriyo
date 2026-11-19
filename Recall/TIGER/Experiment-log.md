# TIGER — Experiment Log

`Recall/TIGER/` 家族所有變體的實驗日誌，最新的日期在最上面；每個條目的 Details 以 **Variant** 標明變體，放在一起方便跨變體比較。超出單一 run 的結論整理到家族的 [INSIGHT.md](INSIGHT.md)；SID 本身的實驗記在 [SID Experiment-log](../../Representations/SID/Experiment-log.md)。

所有 run 都用 2014 Beauty 切分（131,413 train／22,363 validation，歷史上限 20）、seed 42（`train.py` 預設），validation 以全 catalog 的 unconstrained beam search（beam 10）評估。2026-11-19 以前的條目是事後依 log 補記；Expectation 依當時的動機補寫，不參考後來的結果。

## 2026-11-19

### T5X 式 Adafactor 步長（H5）

#### Motivation

同切分的 SASRec validation Recall@10 0.0938，TIGER 不論新舊 SID 都只有 0.034–0.037。程式碼檢查發現 Adafactor 以 `scale_parameter=False` 把 lr 0.01 當成絕對步長，與論文框架 T5X 不同。檢驗 [TIGER INSIGHT](INSIGHT.md) H5。

#### Details

- **Variant:** `baseline`
- **Baseline:** `tiger-paper-v1`（SoC job 1163052，程式 `ea3fb5f`），見下方 2026-11-18 條目。
- **Change:** 只改 `ADAFACTOR_SCALE_PARAMETER=1`（`--adafactor-scale-parameter`，lr 乘上參數 RMS）；SID、模型、資料、seed 都相同。
- **Data:** 2014 Beauty，SID `tiger-paper-v1`（`stable` 配方，256/256/256）。
- **Environment:** SoC job 1165519，節點 xgpf11，程式 `bb1088c`，seed 42，T5 4,815,488 參數。第一次提交的 job 1165195 落在 xgpe2，CUDA 初始化失敗，沒有訓練。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-v1 ADAFACTOR_SCALE_PARAMETER=1 STAGE=train sbatch --export=ALL --job-name=kuriyo-tiger scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/TIGER/baseline/runs/tiger-t5x-adafactor-v1/`；log `slurm-kuriyo-tiger-1165519.out`（repo 根目錄）。

#### Settings

- **資料／前處理：**Amazon Reviews 2014 Beauty 5-core（22,363 users、12,101 items）；每位使用者依 `unixReviewTime` 排序、同時間保留原始列序，最後一筆是 test、倒數第二筆是 validation，其餘每個前綴一筆 train 樣本（131,413 筆）；歷史截到最近 20 件，重複互動保留（`scripts.prepare_amazon_reviews_2014`）。樣本轉成 SID `tiger-paper-v1`（`stable` 配方，三層各用 256 碼；SID 由商品文字的 Sentence-T5 向量產生）。
- **模型：**HF T5，encoder／decoder 各 4 層，d_model 128、d_kv 64、d_ff 1024、6 heads、dropout 0.1、ReLU；詞表為 SID token 加 2,000 個 hashed user token。
- **訓練：**Adafactor，lr 0.01 到 10k steps，之後 0.01·√(10000/step)；batch 256，200,000 steps，歷史上限 20，seed 42。
- **評估：**每 20k steps 在 validation 做 unconstrained beam search（beam 10，eval batch 16），以 NDCG@10 選最佳。
- **Adafactor `scale_parameter`：**True（**與 baseline 唯一的差異**）。
- **參數量：**4,815,488（SID `tiger-paper-v1` 的詞表）。

#### Expectation

若步長是主因，20k、40k 的 validation Recall@10 會明顯高於 baseline 的 0.0132、0.0174，而且收斂所需的 steps 明顯變少。

#### Key Metrics

- validation Recall@10、NDCG@10，step 20k 與 40k。
- 明顯高於 baseline 即支持 H5，接著跑滿 200k；與 baseline 相近則不支持，改查訓練樣本、token 排列與解碼。

#### Results

| Run（step 20k） | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.00939 | 0.01319 | 0.00593 | 0.00718 | `tiger-paper-v1`，job 1163052 |
| T5X Adafactor | 0.05290 | 0.07999 | 0.03467 | 0.04344 | `tiger-t5x-adafactor-v1`，job 1165519 |
| 參考：baseline 跑滿 200k | 0.02182 | 0.03680 | 0.01349 | 0.01835 | 同上 job 1163052 |

| Run | train loss @1k | train loss @1.5k | step 40k Recall@10 |
| --- | --- | --- | --- |
| Baseline | 3.362 | 3.127 | 0.0174 |
| T5X Adafactor | 2.36 | 2.144 | 待定（log 截至 step 30,500） |

**20k 之後（2026-11-19 補記，來自使用者推上的 `events.log`，job 1165519 仍在跑）：**

| step | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | invalid beam |
| --- | --- | --- | --- | --- | --- |
| **20k（best）** | **0.05290** | **0.08000** | **0.03467** | **0.04344** | 0.0006 |
| 40k | 0.04812 | 0.07132 | 0.03163 | 0.03910 | 0.0022 |
| 60k | 0.04114 | 0.06104 | 0.02706 | 0.03349 | 0.0066 |
| 80k | 0.03864 | 0.05710 | 0.02534 | 0.03125 | 0.0125 |
| 100k | 0.03680 | 0.05464 | 0.02433 | 0.03006 | 0.0179 |

- **Observations:** 只改 Adafactor 的縮放，20k 的 validation Recall@10 從 0.0132 變成 0.0800（約 6 倍），NDCG@10 約 6 倍；已是 baseline 跑滿 200k（0.0368）的 2.2 倍，接近同切分 SASRec（0.1023）與 FMRec（0.1031）。**20k 之後每次 validation 都下降**，到 100k 剩 0.0546（−32%）；無效 SID 的 beam 比例從 0.06% 升到 1.8%，下降的速度在 60k 後變慢。`best.pt` 仍是 20k。train loss 在 Git 外（`train.jsonl` 不追蹤，Slurm log 未推），還不知道它是否繼續下降。
- **Hypotheses:** 剩下與 item-ID 模型的差距（約 0.02）可能隨訓練縮小，或來自容量（H4）；未驗證。20k 之後的下降可能是過擬合（train loss 繼續降、validation 降），或 T5X 式步長在 lr 0.01 下仍太大、模型在變差（train loss 也停滯或上升）；最佳點可能在 20k 之前。見 [INSIGHT](INSIGHT.md) H8。

#### Conclusion

強烈支持 H5：Adafactor 步長就是 TIGER 落後的主因。讓 job 繼續跑滿 200k，記錄 40k 之後的曲線與最佳 step；之後的 TIGER 實驗以 `ADAFACTOR_SCALE_PARAMETER=1` 為新 baseline。

**補記（100k）：**20k 之後持續下降，「跑滿 200k」不再必要。下一步是 20k 前後的細曲線（H8），而不是更多 steps。

## 2026-11-18

### 新 SID `tiger-paper-v1` 跑滿 200k（H3）

#### Motivation

舊 SID 在第一個 epoch 發散，三層只用到 6/26/25 個碼。檢驗 [TIGER INSIGHT](INSIGHT.md) H3：SID 品質是與論文差距的主因。

#### Details

- **Variant:** `baseline`
- **Baseline:** `tiger-paper-20261117-005232`（job 1153307，舊 SID），見 2026-11-17 條目。
- **Change:** 改用 `stable` 配方的新 SID `tiger-paper-v1`（[SID Experiment-log](../../Representations/SID/Experiment-log.md) 2026-11-18，job 1162430）；T5 設定不變。新 SID 的碰撞 token 較少，詞表較小，T5 參數由 5,587,968 變成 4,815,488。
- **Data:** 2014 Beauty，SID `tiger-paper-v1`。
- **Environment:** SoC job 1163050（`STAGE=materialize`）與 job 1163052（`STAGE=train`），程式 `ea3fb5f`，seed 42。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-paper-v1 STAGE=train sbatch --export=ALL scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/TIGER/baseline/runs/tiger-paper-v1/`；log 已從工作目錄移除，用 `git show b7a4078:slurm-kuriyo-tiger-1163052.out` 取回。

#### Settings

- **資料／前處理：**Amazon Reviews 2014 Beauty 5-core（22,363 users、12,101 items）；每位使用者依 `unixReviewTime` 排序、同時間保留原始列序，最後一筆是 test、倒數第二筆是 validation，其餘每個前綴一筆 train 樣本（131,413 筆）；歷史截到最近 20 件，重複互動保留（`scripts.prepare_amazon_reviews_2014`）。樣本轉成 SID `tiger-paper-v1`（`stable` 配方，三層各用 256 碼；SID 由商品文字的 Sentence-T5 向量產生）。
- **模型：**HF T5，encoder／decoder 各 4 層，d_model 128、d_kv 64、d_ff 1024、6 heads、dropout 0.1、ReLU；詞表為 SID token 加 2,000 個 hashed user token。
- **訓練：**Adafactor，lr 0.01 到 10k steps，之後 0.01·√(10000/step)；batch 256，200,000 steps，歷史上限 20，seed 42。
- **評估：**每 20k steps 在 validation 做 unconstrained beam search（beam 10，eval batch 16），以 NDCG@10 選最佳。
- **Adafactor `scale_parameter`：**False。
- **參數量：**4,815,488（新 SID 詞表較小）。

#### Expectation

換成全碼本使用的 SID 後，validation Recall@10 會明顯高於舊 SID 的 0.0342。

#### Key Metrics

- validation Recall@10、NDCG@10，先比 20k／60k，再看 200k。
- 明顯高於 0.0342（超出標準誤約 0.0013 數倍）才算支持 H3。

#### Results

| step | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| 20k | 0.00939 | 0.01319 | 0.00593 | 0.00718 |
| 40k | 0.01105 | 0.01744 | 0.00696 | 0.00901 |
| 60k | 0.01391 | 0.02307 | 0.00872 | 0.01176 |
| 80k | 0.01614 | 0.02714 | 0.00979 | 0.01336 |
| 100k | 0.01596 | 0.02848 | 0.01005 | 0.01414 |
| 120k | 0.01672 | 0.03099 | 0.01058 | 0.01527 |
| 140k | 0.01851 | 0.03036 | 0.01134 | 0.01522 |
| 160k | 0.01887 | 0.03242 | 0.01179 | 0.01622 |
| 180k | 0.02066 | 0.03613 | 0.01266 | 0.01766 |
| **200k** | **0.021822** | **0.036801** | **0.013492** | **0.018354** |
| 舊 SID 200k | 0.020570 | 0.034208 | 0.012409 | 0.016788 |

- **Observations:** 200k 只比舊 SID 高 0.0026 Recall@10；曲線到 200k 仍在上升。invalid_beam_fraction 每次都是 0，最終 train loss 1.694。新 SID 第一層碼分布較平均，每個 token 不確定性較高，兩個 run 的 loss 不能直接比。
- **Hypotheses:** 差距較可能在序列模型的訓練（後來寫成 H5）、訓練樣本或解碼。

#### Conclusion

不支持 H3。保留 `tiger-paper-v1` 作為之後 TIGER 實驗的 baseline；下一步是 SASRec 同切分檢查與 H5。沒有跑 test。

## 2026-11-17

### 首次完整 paper-configured run（job 1153307）

#### Motivation

在完整 Beauty 上跑出第一個依論文公開設定的 TIGER，作為之後所有比較的起點。

#### Details

- **Variant:** `baseline`
- **Baseline:** 無（首次完整 run）；比較目標是 TIGER 論文的 Beauty test 數字（Recall@10 0.0648）。
- **Change:** —
- **Data:** 2014 Beauty，SID 與 RUN_NAME 都是 `tiger-paper-20261117-005232`（`paper-pytorch` 行為，舊 SID）。
- **Environment:** SoC job 1153307，程式 `5f99955`，seed 42，T5 5,587,968 參數。
- **Command:** `SID_VERSION=tiger-paper-20261117-005232 RUN_NAME=tiger-paper-20261117-005232 STAGE=all sbatch --export=ALL scripts/run.sbatch`（依 log 記錄的設定重建；SID 20,000 epochs → materialize → TIGER 200,000 steps）
- **Artifacts:** SoC 舊路徑 `Recall/TIGER/runs/tiger-paper-20261117-005232/`（續訓或 test 前需搬到 `baseline/runs/`）；log 用 `git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out` 取回；本機匯入紀錄在 `runs/tiger-paper-20261117-005232/imports/slurm-1153307/`（重建指令見 [HANDOFF](../../HANDOFF.md)）。

#### Settings

- **資料／前處理：**Amazon Reviews 2014 Beauty 5-core（22,363 users、12,101 items）；每位使用者依 `unixReviewTime` 排序、同時間保留原始列序，最後一筆是 test、倒數第二筆是 validation，其餘每個前綴一筆 train 樣本（131,413 筆）；歷史截到最近 20 件，重複互動保留（`scripts.prepare_amazon_reviews_2014`）。樣本轉成 job 1153307 的舊 SID（論文設定的 PyTorch 實作，三層只用 6/26/25 碼）。
- **模型：**HF T5，encoder／decoder 各 4 層，d_model 128、d_kv 64、d_ff 1024、6 heads、dropout 0.1、ReLU；詞表為 SID token 加 2,000 個 hashed user token。
- **訓練：**Adafactor，lr 0.01 到 10k steps，之後 0.01·√(10000/step)；batch 256，200,000 steps，歷史上限 20，seed 42。
- **評估：**每 20k steps 在 validation 做 unconstrained beam search（beam 10，eval batch 16），以 NDCG@10 選最佳。
- **Adafactor `scale_parameter`：**False（當時入口 `Recall/TIGER/scripts/train_paper_baseline.py`，設定與現行 `train.py` 相同）。
- **參數量：**5,587,968。

#### Expectation

依論文公開設定實作，validation Recall@10 應與論文 Beauty 的 0.0648 同一量級（validation 與 test 同協定、只差最後一筆），至少明顯高於熱門商品基線。若遠低於論文，代表實作或設定和論文有落差，需要逐項排查。

#### Key Metrics

- validation Recall@5/10、NDCG@5/10，每 20k steps 一次，以 NDCG@10 選最佳。

#### Results

| Run | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Notes |
| --- | --- | --- | --- | --- | --- |
| step 200k（最佳） | 0.020570 | 0.034208 | 0.012409 | 0.016788 | `tiger-paper-20261117-005232` |
| 論文（test） | 0.0454 | 0.0648 | 0.0321 | 0.0384 | 不同 split，僅作參考 |

- **Observations:** 十次 validation 中最佳 NDCG@10 在最後一次（200k），曲線不是嚴格單調；invalid_beam_fraction 都是 0。log 最後印出 Saved；沒有 sacct 成功狀態或 checkpoint 校驗。SID 三層只用 6/26/25 碼（見 SID Experiment-log）。
- **Hypotheses:** SID 塌縮可能是差距主因（後來寫成 H3）；模型容量只有論文約 1,300 萬的一半以下（H4）。

#### Conclusion

取得第一個可比較的 baseline，validation Recall@10 0.0342，約為熱門商品基線（0.0170）的 2 倍，遠低於論文。下一步先查 SID。沒有跑 test。
