# FMRec — Experiment Log

`Recall/FMRec/` 家族所有變體的實驗日誌，最新的日期在最上面；每個條目的 Details 以 **Variant** 標明變體，放在一起方便跨變體比較。超出單一 run 的結論整理到家族的 [INSIGHT.md](INSIGHT.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

所有 run 都用 2014 Beauty ASIN 切分（131,413 train／22,363 validation，歷史上限 20），validation 對全 catalog 12,101 件排名、不過濾歷史，以 NDCG@10 選最佳。

## 2026-11-21

### 作者預設 5 個 seed（42–46）的 validation

#### Motivation

FMRec 是 Kuriyo 與論文最接近的結果，但只有 seed 42。README「實驗設計與統計檢定」規定與論文並列的數字要 5 個 seed；同時以共同的 seed 42–44 和 SASRec CE 做配對與 seed 層級比較。

#### Details

- **Variant:** `baseline`
- **Baseline:** `fmrec-beauty-v1`（seed 42，2026-11-19 條目）。
- **Change:** 只改 seed（43、44、45、46）；設定全部是作者預設。
- **Data:** 2014 Beauty ASIN 切分；validation 22,363 筆。
- **Environment:** SoC jobs 1168952–54（s43–s45，程式 `7209f60`）、1170268（s46，`e92cd50`）；兩個 revision 之間 FMRec 的訓練與評估程式沒有改動（中間只有紀錄與 TIGER 旗標）。
- **Command:** `MODEL=fmrec RUN_NAME=fmrec-beauty-s<seed> SEED=<seed> STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`
- **Artifacts:** `Recall/FMRec/baseline/runs/fmrec-beauty-s{43,44,45,46}/`（含 `best-ranks.json`）。

#### Settings

- 與 `fmrec-beauty-v1` 相同（作者預設：hidden 128、4 blocks、Euler 30 步、lr 1e-3、batch 512、每 20 epochs 評估、patience 5），**不同**：`run.seed`。

#### Expectation

事後記錄：提交時沒有寫下數值預期。

#### Key Metrics

- 5 個 seed 的 validation Recall@5/10、NDCG@5/10 平均 ± 標準差與 t 95% CI；與 SASRec CE（只換 CE，seed 42–44）的配對與 Welch 檢定。

#### Results

| seed | best epoch | Recall@10 | NDCG@10 |
| --- | --- | --- | --- |
| 42（`fmrec-beauty-v1`） | 160 | 0.10307 | 0.06035 |
| 43 | 160 | 0.10356 | 0.06094 |
| 44 | 160 | 0.10508 | 0.06162 |
| 45 | 160 | 0.10428 | 0.06072 |
| 46 | 360 | 0.10075 | 0.05718 |
| **平均 ± sd（95% CI）** | | **0.10335 ± 0.00164（0.10131–0.10539）** | **0.06016 ± 0.00173（0.05801–0.06231）** |

Recall@5 0.07285 ± 0.00207，NDCG@5 0.05033 ± 0.00187。

`python -m core.compare`，seed 42–44，對 SASRec 只換 CE（`sasrec-paperce-beauty-*`）：

| 指標 | SASRec CE | FMRec | 差 | 配對 t p | bootstrap 95% CI | Welch p | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recall@10 | 0.10237 ± 0.00061 | 0.10391 ± 0.00105 | +0.00154 | 0.2 | −0.0008 到 +0.0039 | 0.11 | 沒有差異 |
| NDCG@10 | 0.05072 ± 0.00042 | 0.06097 ± 0.00064 | +0.01025 | 7e-47 | +0.0089 到 +0.0117 | 6e-5 | 顯著較好 |

- **Observations:** seed 42–45 都在 epoch 160 最好、260 early stop；seed 46 在 360 最好、460 停，數字最低。NDCG@10 的優勢遠大於 seed 標準差，Recall@10 沒有差異。
- **Hypotheses:** FMRec 的優勢在把目標排得更前面，而不是命中更多（與 2026-11-20 單一 seed 的配對結果一致）。seed 46 的訓練路徑不同（較晚才到最佳），原因未查。

#### Conclusion

FMRec 作者預設在 Kuriyo 切分的 validation：Recall@10 0.1034 ± 0.0016、NDCG@10 0.0602 ± 0.0017（5 seeds）。相對 SASRec CE，NDCG@10 顯著較好、Recall@10 沒有差異。下一步：seed 43–46 各跑一次 test，報告 5 個 seed 的 test 平均 ± 標準差與論文並列。

## 2026-11-20

### 分組重新評估與逐使用者排名（`STAGE=groups`）

#### Motivation

依目標熱門度拆解整體指標（`Recall/INSIGHT.md` H2），並補出逐使用者排名，和 SASRec、TIGER 做配對比較。

#### Details

- **Variant:** `baseline`
- **Baseline:** `fmrec-beauty-v1` 的原紀錄。
- **Change:** 無；只重新評估（評估雜訊以固定 seed 產生，所以可重現）。
- **Data:** 同 SASRec 條目（2014 Beauty，validation／test 各 22,363 筆）。
- **Environment:** SoC job 1168948，程式 `7209f60`。
- **Command:** `MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=groups sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`
- **Artifacts:** `Recall/FMRec/baseline/runs/fmrec-beauty-v1/` 的 `validation/amazon2014-beauty/best-groups.json`、`best-ranks.json`、`test/amazon2014-beauty/groups.json`、`ranks.json`。

#### Settings

- 不訓練；以各 run 的 `checkpoints/best.pt` 重新評估，評估設定與原 run 相同（`settings.json`），只多分組與逐使用者排名。分組依該 run 實際訓練的 train。

#### Expectation

在 `Recall/INSIGHT.md` H2 與 TIGER H7 中事先寫下：若內容 ID 有冷門優勢，TIGER 與 item-ID 模型的差距會在 1–4 組與 0 組縮小或反轉。

#### Key Metrics

- 各熱門度組（目標在 train 當目標 0、1–4、5 次以上）的 Recall@10／NDCG@10；整體數字與原紀錄的差（`max_abs_difference`）；逐使用者配對檢定。

#### Results

| split | 1–4 R@10 | 1–4 N@10 | 5+ R@10 | 5+ N@10 |
| --- | --- | --- | --- | --- |
| validation | 0.0344 | 0.0213 | 0.1212 | 0.0707 |
| test | 0.0283 | 0.0168 | 0.0984 | 0.0581 |

- **Observations:** `max_abs_difference` 為 0；0 組全為 0。對 SASRec CE（`sasrec-beauty-v1`）的逐使用者配對：NDCG@10 validation +0.0095、test +0.0061（皆 p < 1e-12）；Recall@10 validation +0.0008（p = 0.62）、test −0.0047（p = 0.002）。validation → test 兩組各降 18%／19%。
- **Hypotheses:** FMRec 與 SASRec 命中前 10 的比例相當，但把目標排得更前面（NDCG 較高），兩組都如此。單一 seed，不寫「顯著較好」。

#### Conclusion

FMRec 在冷門組是三個模型中最強（1–4 組 NDCG@10 0.021，SASRec 0.014），整體優勢在 NDCG 而非 Recall。等 seed 43–46 完成後用 5 個 seed 重做配對與 Welch 檢定。

## 2026-11-19

### Held-out test：`fmrec-beauty-v1`（seed 42）

#### Motivation

`fmrec-beauty-v1` 已訓練完並以 validation 選定 `best.pt`；先取得一次 test 數字，與論文的 test 並列。之後的設定比較仍只看 validation。

#### Details

- **Variant:** `baseline`
- **Baseline:** 同一 run 的 validation（下方 SoC 條目）。
- **Change:** 無；只對已選定的 `best.pt` 評估 test 一次。
- **Data:** 2014 Beauty ASIN 切分的 test（22,363 筆，歷史含 validation 商品），test sha256 `88e1d637…`。
- **Environment:** SoC，程式 `4f6594d`（舊平版面在這次 test 時搬進新版面）；job ID 未記錄（不影響結果；之後重送的 job 1166757 因「test 已跑過」被拒絕）。2026-11-19 之後的 test 會把 Git revision 與 job ID 寫進 `metrics.json`。
- **Command:** `MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=test sbatch --export=ALL --exclude='xgpe[0-4,6]' --job-name=kuriyo-fmrec scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/FMRec/baseline/runs/fmrec-beauty-v1/test/amazon2014-beauty/metrics.json`。

#### Settings

- 與下方 SoC 條目相同（設定記在該 run 的 `settings.json`）。

#### Expectation

事後補記（test 跑完才寫）：test 低於 validation，因為 test 目標的冷門與長尾比例較高（`Recall/INSIGHT.md` Facts）。作者預設設定。

#### Key Metrics

- test Recall@5/10、NDCG@5/10，與 validation 及論文 test 的差距。

#### Results

| | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| validation（best） | 0.07280 | 0.10307 | 0.05058 | 0.06035 |
| **test** | **0.05764** | **0.08036** | **0.04017** | **0.04745** |
| FMRec 論文（HR@K，50 件歷史、ICLRec 處理的資料） | 0.0584 | 0.0827 | 0.0416 | 0.0495 |

- **Observations:** test Recall@10 比 validation 低 0.0227（約 22%），NDCG@10 低 0.0129。test 的四個指標都在論文數字的 0.0025 以內（Recall@10 低 0.0023），但論文用 50 件歷史與另一份處理的資料，只能當參考。和 SASRec CE 的 test 相比，Recall@10 低 0.0047、NDCG@10 高 0.0061，都在單一 seed 能分辨的範圍附近。
- **Hypotheses:** 單一 seed；使用者層級的標準誤約 0.0019（Recall@10），與其他模型的差距在這個範圍內不能下結論（README「實驗設計與統計檢定」）。

#### Conclusion

作者預設 seed 42 的 test 是 Recall@10 0.0804、NDCG@10 0.0474，與論文報告的數字接近。這個 run 不再訓練或調整。

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
