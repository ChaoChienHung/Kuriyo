# TIGER — Experiment Log

`Recall/TIGER/` 家族所有變體的實驗日誌，最新的日期在最上面；每個條目的 Details 以 **Variant** 標明變體，放在一起方便跨變體比較。超出單一 run 的結論整理到家族的 [INSIGHT.md](INSIGHT.md)；SID 本身的實驗記在 [SID Experiment-log](../../Representations/SID/Experiment-log.md)。

所有 run 都用 2014 Beauty 切分（131,413 train／22,363 validation，歷史上限 20）、seed 42（`train.py` 預設），validation 以全 catalog 的 unconstrained beam search（beam 10）評估。2026-11-19 以前的條目是事後依 log 補記；Expectation 依當時的動機補寫，不參考後來的結果。

## 2026-11-21

### Dropout 0.2：`tiger-t5x-adafactor-dropout02-v1`（H8）

#### Motivation

2k 間隔曲線（下方條目）顯示 20k 之後對熱門商品過擬合：train loss 持續下降、5+ 組 Recall@10 從 0.0975 降到 0.0858，1–4 組仍上升。加強正則化看能否延後或壓低下降，並讓冷門組繼續進步（[INSIGHT](INSIGHT.md) H8、H7）。

#### Details

- **Variant:** `baseline`（`--dropout` 是 2026-11-20 加入的選擇性旗標，預設 0.1 不變）
- **Baseline:** `tiger-t5x-adafactor-eval2k-v1`（同 seed、同 SID、同評估間隔）。
- **Change:** T5 `dropout_rate` 0.1 → 0.2，其他不變。
- **Data:** 2014 Beauty SID `tiger-paper-v1`；validation 22,363 筆。
- **Environment:** SoC job 1175903，程式 `da61807`（`settings.json`）。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-dropout02-v1 TRAIN_ARGS="--dropout 0.2 --eval-every 2000 --steps 40000" STAGE=train sbatch --export=ALL --job-name=kuriyo-tiger scripts/run.sbatch`
- **Artifacts:** `Recall/TIGER/baseline/runs/tiger-t5x-adafactor-dropout02-v1/`。

#### Settings

- 與 baseline 相同，**不同**：`model.config.dropout_rate` 0.2。

#### Expectation

run 前寫下（2026-11-20）：同 step 的 train loss 高於 baseline；validation 最佳點晚於 20k（約 24k–34k），最佳 Recall@10 與 baseline 相當或略高（0.078–0.084），最佳點之後的下降小於 baseline（40k 高於 0.0713）；最佳點的 1–4 組 Recall@10 不低於 0.0134。若最佳值低於 0.076，代表 0.2 已經欠擬合。單一 seed，差距在 ±0.003 內不下結論。

#### Key Metrics

- 每 2k step 的 validation Recall@10／NDCG@10、無效 beam；1–4 組與 5+ 組 Recall@10；train loss；最佳 step。

#### Results

單一 seed（42）。Validation 每 2k step（Recall@10；分組為 1–4／5+ 組 Recall@10），與 baseline 並列：

| step | dropout 0.2 | 1–4 | 5+ | baseline 0.1 | 1–4 | 5+ |
|---|---|---|---|---|---|---|
| 10k | 0.0627 | 0.0004 | 0.0791 | 0.0739 | 0.0066 | 0.0916 |
| 20k | 0.0751 | 0.0077 | 0.0928 | **0.0800** | 0.0134 | 0.0975 |
| 26k | 0.0795 | 0.0079 | 0.0983 | 0.0771 | 0.0125 | 0.0941 |
| 30k | 0.0787 | 0.0074 | 0.0974 | 0.0759 | 0.0129 | 0.0924 |
| 34k | 0.0805 | 0.0088 | 0.0993 | 0.0729 | 0.0167 | 0.0878 |
| 38k | **0.0805** | 0.0077 | 0.0996 | 0.0702 | 0.0138 | 0.0851 |
| 40k | 0.0802 | 0.0085 | 0.0990 | 0.0713 | 0.0162 | 0.0858 |

- **最佳點：**38k，Recall@10 0.08049、NDCG@10 0.04420、Recall@5 0.05362，無效 beam 0.085%（`best.json`）。34k–40k 是平台（0.0802–0.0805），40k 時仍未下降。
- **學習較慢：**26k 之前每個 step 都低於 baseline（10k 低 0.011），約 26k 交叉；之後 baseline 下降，dropout 0.2 持平。
- **對 baseline 最佳點（20k）的逐使用者配對（`python -m core.compare`，22,363 位使用者）：**Recall@10 +0.0005（paired t p 0.76，bootstrap 95% CI [−0.0027, +0.0037]）；NDCG@10 +0.0008（p 0.42，CI [−0.0010, +0.0026]）。單一 seed，不做 seed 層級檢定。
- **分組：**最佳點的 1–4 組 Recall@10 0.0077（4,564 筆中 35 筆命中），baseline 20k 0.0134（61 筆）；1–4 組在每個 step 都低於 baseline，最高 0.0092（36k）。5+ 組 0.0996，baseline 20k 0.0975。0 組仍為 0。
- **Train loss**（每點為結束於該 step 的 1,000 步平均，取自 SoC 上兩個 run 的 `train/train.jsonl`，使用者 2026-11-21 回報）：dropout 0.2 為 10k 1.845、20k 1.687、30k 1.608、40k 1.549；baseline 為 1.631、1.442、1.290、1.158。dropout 0.2 在每個 step 都較高，40k 仍在下降，約等於 baseline 15k 的水準。訓練時的 loss 含 dropout 雜訊，兩者不完全可比。
- **對照 Expectation：**最佳點晚於 20k（是，38k，晚於預期的 24k–34k）；最佳 Recall@10 在 0.078–0.084（是）；40k 高於 0.0713（是，0.0802）；1–4 組不低於 0.0134（否，0.0077）；沒有欠擬合（最佳值高於 0.076）；同 step 的 train loss 較高（是）。

#### Conclusion

Dropout 0.2 把 20k 之後的下降推遲到至少 40k，但沒有提高最佳值：差距 +0.0005 在事先定的 ±0.003 內，配對檢定也不顯著（單一 seed）。最佳值的提升與損失都來自熱門組；冷門（1–4）組反而更差，推測 0.1 dropout 下冷門組在 20k 後的上升是靠記憶，加強正則化會抑制它。過擬合決定的是最佳點出現在哪一步，不是 TIGER 約 0.080 的上限，所以更強的 dropout 或 weight decay 不太可能補上與 SASRec／FMRec（約 0.10）的差距；下一步改查容量與資料（H4）。0.1 dropout 的 20k 仍是較省的設定（一半的 step、相同的最佳值）。

## 2026-11-20

### 分組重新評估與逐使用者排名：`tiger-t5x-adafactor-v1`（H7）

#### Motivation

檢驗 [INSIGHT](INSIGHT.md) H7（SID 讓 TIGER 在冷門商品上占優），並找出 TIGER validation → test 下降 31%（多於 item-ID 模型）的來源。

#### Details

- **Variant:** `baseline`
- **Baseline:** `tiger-t5x-adafactor-v1` 的原紀錄；跨模型比較對 SASRec CE（`sasrec-beauty-v1`）與 FMRec（`fmrec-beauty-v1`）。
- **Change:** 無；只對 20k 的 `best.pt` 重新評估。
- **Data:** 2014 Beauty SID `tiger-paper-v1`；validation／test 各 22,363 筆。
- **Environment:** SoC job 1168949，程式 `7209f60`。`tiger-t5x-adafactor-eval2k-v1` 的分組 job 尚未執行（訓練時的每個 validation 已帶分組）。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-v1 STAGE=groups sbatch --export=ALL --job-name=kuriyo-tiger scripts/run.sbatch`
- **Artifacts:** `Recall/TIGER/baseline/runs/tiger-t5x-adafactor-v1/` 的 `validation/amazon2014-beauty/best-groups.json`、`best-ranks.json`、`test/amazon2014-beauty/groups.json`、`ranks.json`。

#### Settings

- 不訓練；以各 run 的 `checkpoints/best.pt` 重新評估，評估設定與原 run 相同（`settings.json`），只多分組與逐使用者排名。分組依該 run 實際訓練的 train。
- beam 10、unconstrained，無效 SID 算未命中（同原 run）。

#### Expectation

在 `Recall/INSIGHT.md` H2 與 TIGER H7 中事先寫下：若內容 ID 有冷門優勢，TIGER 與 item-ID 模型的差距會在 1–4 組與 0 組縮小或反轉。

#### Key Metrics

- 各熱門度組（目標在 train 當目標 0、1–4、5 次以上）的 Recall@10／NDCG@10；整體數字與原紀錄的差（`max_abs_difference`）；逐使用者配對檢定。

#### Results

| split | 1–4 R@10 | 1–4 N@10 | 5+ R@10 | 5+ N@10 | 1–4 相對 SASRec／FMRec | 5+ 相對 SASRec／FMRec |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 0.0134 | 0.0060 | 0.0975 | 0.0533 | 0.44／0.39 | 0.80／0.80 |
| test | 0.0067 | 0.0024 | 0.0723 | 0.0386 | 0.23／0.24 | 0.69／0.73 |

- **Observations:** `max_abs_difference` 為 0；0 組（validation 81、test 196）全為 0，item-ID 模型也是 0。以 validation 的分組 Recall@10 套用 test 的組成得 0.0760，只解釋 0.0800 → 0.0556 下降的 16%；其餘是組內下降：1–4 組 −50%、5+ 組 −26%（SASRec −2%／−14%，FMRec −18%／−19%）。逐使用者配對對 SASRec CE：Recall@10 validation −0.022、test −0.029，NDCG@10 −0.007／−0.012（皆 p < 1e-12，單一 seed）。
- **Hypotheses:** 目前的 TIGER 偏向熱門商品：beam search 從熱門前綴展開，冷門商品的 SID 很少進入前 10。test 的組內下降可能與 20k 已開始對熱門商品過擬合（H8）有關；2k 曲線中 1–4 組在 20k 後仍上升。

#### Conclusion

H7 不支持：TIGER 在冷門組的差距最大，不是最小；test 的額外下降主要是組內，尤其冷門組減半。下一步的正則化實驗（H8，TODO）同時看 1–4 組；之後可試 constrained decoding 或調整 beam，看冷門 SID 是否被 beam 剪掉（新變體）。

### 2k 間隔的細曲線：`tiger-t5x-adafactor-eval2k-v1`（H8）

#### Motivation

job 1165519 每 20k 才評估一次，只知道 20k（0.0800）比 40k（0.0713）好。用 2k 間隔找出最佳 step，並確認 20k 後的下降是過擬合（[INSIGHT](INSIGHT.md) H8）。

#### Details

- **Variant:** `baseline`
- **Baseline:** `tiger-t5x-adafactor-v1`（job 1165519，同 seed、同 SID，每 20k 評估）。
- **Change:** 只改評估間隔與總步數（`--eval-every 2000 --steps 40000`）；模型、optimizer、學習率排程不變。
- **Data:** 2014 Beauty SID `tiger-paper-v1`；validation 22,363 筆。
- **Environment:** SoC job 1167955，程式 `2b19879`（記在 `settings.json` 與 `events.log`）；第一次提交 job 1167529 因 Slurm controller 連線失敗沒有訓練。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-eval2k-v1 TRAIN_ARGS="--eval-every 2000 --steps 40000" STAGE=train sbatch --export=ALL scripts/run.sbatch`
- **Artifacts:** `Recall/TIGER/baseline/runs/tiger-t5x-adafactor-eval2k-v1/`（`validation/amazon2014-beauty/step-*.json` 含分組指標）；train loss 取自 Slurm log，`git show 9eca60c:slurm-kuriyo-tiger-1167955.out`。

#### Settings

- 與 `tiger-t5x-adafactor-v1` 相同（T5X 式 Adafactor、lr 0.01 前 10k 固定後 inverse sqrt、batch 256、seed 42），**不同**：`evaluation.every_steps` 2,000、`training.steps` 40,000。學習率排程與總步數無關，所以前 40k 的訓練與 baseline 相同。

#### Expectation

事後記錄：TODO 寫的是「找出最佳 step」，沒有寫數值預期；run 進行到 16k 時在對話中估計最佳點在 14k–22k 之間。

#### Key Metrics

- 各 step 的 validation Recall@10／NDCG@10、無效 beam 比例；train loss；目標熱門度分組（train 當目標 1–4 次、5 次以上）的 Recall@10。

#### Results

| step | Recall@10 | NDCG@10 | 無效 beam | 1–4 組 R@10 | 5+ 組 R@10 | train loss |
| --- | --- | --- | --- | --- | --- | --- |
| 4k | 0.06028 | 0.02983 | 0.00% | 0.0011 | 0.0758 | 1.82 |
| 8k | 0.07190 | 0.03659 | 0.02% | 0.0053 | 0.0894 | 1.70 |
| 12k | 0.07401 | 0.03915 | 0.03% | 0.0083 | 0.0913 | 1.61 |
| 16k | 0.07749 | 0.04131 | 0.04% | 0.0092 | 0.0954 | 1.50 |
| **20k（best）** | **0.08000** | **0.04344** | 0.06% | 0.0134 | 0.0975 | 1.42 |
| 24k | 0.07562 | 0.04194 | 0.05% | 0.0123 | 0.0923 | 1.42 |
| 28k | 0.07647 | 0.04167 | 0.11% | 0.0131 | 0.0931 | 1.37 |
| 32k | 0.07307 | 0.03985 | 0.15% | 0.0149 | 0.0884 | 1.29 |
| 36k | 0.07021 | 0.03834 | 0.21% | 0.0123 | 0.0854 | 1.21 |
| 40k | 0.07132 | 0.03910 | 0.22% | 0.0162 | 0.0858 | 1.16 |

train loss 是 Slurm log 該 step 的單一 batch 值。從未當過 train 目標的 81 筆在每個 step 都是 0。

- **Observations:** 最佳 step 是 20k，Recall@10 0.08000、NDCG@10 0.04344，與 job 1165519 的 20k 逐位相同（同 seed 可重現）；40k 為 0.07132，也與 job 1165519 的 40k（0.0713）一致。14k–22k 是平台（0.0775–0.0800），之後下降，train loss 從 1.42 持續降到 1.16，無效 beam 從 0.06% 升到 0.22%。分組走向不同：5+ 組在 20k 最高後下降（0.0975 → 0.0858），1–4 組大致持續上升（0.0134 → 0.0162，34k 為 0.0167）。
- **Hypotheses:** 20k 之後 5+ 組下降、train loss 下降，是對熱門商品的過擬合；1–4 組仍在學，代表冷門商品需要更多步數，過擬合不是全面的。單一 seed，1–4 組 4,564 筆，相鄰 step 的差距（約 0.002）在雜訊內。

#### Conclusion

H8 的最佳 step 確定在 20k（平台 14k–22k），20k 後是過擬合。以 NDCG@10 選擇時，`tiger-t5x-adafactor-v1` 的 20k 就是最佳點，不必重跑 test。下一步的單一因素是正則化（dropout 0.1 → 0.2，或 weight decay），看能否同時保住 5+ 組並讓 1–4 組繼續上升；改設定用新 RUN_NAME。

## 2026-11-19

### Held-out test：`tiger-t5x-adafactor-v1`（20k，seed 42）

#### Motivation

確認 TIGER 與論文 test 的實際差距。job 1165519 已結束，`best.pt`（20k）不會再變，測試它不影響之後的選擇。

#### Details

- **Variant:** `baseline`
- **Baseline:** 同一 run 的 validation（下方 T5X 式 Adafactor 條目）。
- **Change:** 無；只對 20k 的 `best.pt` 評估 test 一次。
- **Data:** 2014 Beauty SID `tiger-paper-v1` 的 test 切分（22,363 筆）。
- **Environment:** SoC job 1167548（記在 `metrics.json`）。舊平版面在這次 test 時搬進新版面。
- **Command:** `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-v1 STAGE=test sbatch --export=ALL --exclude='xgpe[0-4,6]' scripts/run.sbatch`
- **Artifacts:** `Recall/TIGER/baseline/runs/tiger-t5x-adafactor-v1/test/amazon2014-beauty/metrics.json`。

#### Settings

- 與下方 T5X 式 Adafactor 條目相同（`settings.json`）。

#### Expectation

在結果出來前於對話中估計、事後記錄：依 SASRec／FMRec 的 test／validation 比例（−17%／−22%），test Recall@10 約 0.062–0.066、NDCG@10 約 0.034–0.036，與論文（0.0648／0.0384）相當。

#### Key Metrics

- test Recall@5/10、NDCG@5/10；test 相對 validation 的下降幅度，與 SASRec、FMRec 比。

#### Results

| | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| validation（20k） | 0.05290 | 0.08000 | 0.03467 | 0.04344 |
| **test** | **0.03564** | **0.05558** | **0.02298** | **0.02942** |
| TIGER 論文（test） | 0.0454 | 0.0648 | 0.0321 | 0.0384 |

- **Observations:** test Recall@10 比 validation 低 30.5%，NDCG@10 低 32.3%，明顯多於 SASRec CE（−17%）與 FMRec（−22%）；比論文低 0.0092 Recall@10（約 −14%）、0.0094 NDCG@10。invalid_beam_fraction 0.0003，不是無效 SID 造成。Expectation 不成立。
- **Hypotheses:** test 目標的長尾比例較高（train 出現少於 5 次：test 25.4%、validation 20.8%）；若 TIGER 在冷門目標上比 item-ID 模型差，就會掉得更多，這與 [INSIGHT](INSIGHT.md) H7 的預期相反。20k 是過擬合曲線上的點（H8），細曲線找到的較早 step 可能泛化較好。單一 seed。

#### Conclusion

TIGER 在 test 上仍低於論文約 14%，不是先前以為的「已達論文水準」。下一步：依目標熱門度分組評估（TODO），看 TIGER 掉在哪一組；以及 2k 間隔的細曲線（H8）。

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

**補記（train loss，Slurm log `git show 5963e3f:slurm-kuriyo-tiger-1165519.out`，只到 step 30,500）：**step 1 loss 8.898、10,000 1.599、20,000 1.416、30,000 1.297。使用者另外取樣：step 4,900 loss 1.803（lr 0.01）、9,900 1.625、14,900 1.473（lr 0.0082）、19,900 1.495、24,900 1.388、29,900 1.286（lr 0.0058）。validation 在 20k 之後下降時 train loss 仍在降，支持過擬合（H8）。每 epoch 約 513 steps（131,413 樣本／batch 256），20k 約 39 epochs、200k 約 390 epochs。使用者在 100k 之後手動結束 job 1165519；`best.pt` 是 20k。

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
