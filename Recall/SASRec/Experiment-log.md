# SASRec — Experiment Log

`Recall/SASRec/` 家族所有變體的實驗日誌，最新的日期在最上面；每個條目的 Details 以 **Variant** 標明變體，放在一起方便跨變體比較。超出單一 run 的結論整理到家族的 [INSIGHT.md](INSIGHT.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

2026-11-19 起 baseline 的預設值改成原論文設定（BCE、負例排除使用者看過的商品、hidden 50／1 head）。以下的既有條目都是改動前的預設，也就是 `--loss ce --hidden 64 --heads 2` 這組設定；新條目在 Settings 寫明用了哪組設定。

所有 run 都用 2014 Beauty ASIN 切分（131,413 train／22,363 validation，歷史上限 20），validation 對全 catalog 12,101 件排名、不過濾歷史，以 NDCG@10 選最佳。

## 2026-11-21

### H1 三個 seed 的配對檢定（BCE 對 CE）

#### Motivation

2026-11-19 的 H1 結論只有 seed 層級的 Welch 檢定；README 的主要檢定是逐使用者配對。補齊 `sasrec-paper-beauty-s43`／`-s44`、`sasrec-paperce-beauty-s44` 的排名後做完整檢定。

#### Details

- **Variant:** `baseline`
- **Baseline:** 論文設定（BCE）3 個 seed。
- **Change:** 無新訓練；三個 run 的 `STAGE=groups` 補出排名（程式 `36336f3` 之前的 main；job ID 記在各 `best-groups.json`），再以 `python -m core.compare` 比較。
- **Data:** 2014 Beauty validation 22,363 筆。
- **Environment:** 本機執行 `core.compare`（bootstrap 10,000 次、seed 0）。
- **Command:** `python -m core.compare --metric recall@10 --setting bce Recall/SASRec/baseline/runs/sasrec-paper-beauty-{v1,s43,s44} --setting ce Recall/SASRec/baseline/runs/sasrec-paperce-beauty-{v1,s43,s44}`（NDCG@10 同理）
- **Artifacts:** 六個 run 的 `validation/amazon2014-beauty/best-ranks.json`、`best-groups.json`。

#### Settings

- 同 2026-11-19「論文設定 vs 只換 CE（H1）」條目。

#### Expectation

2026-11-19 條目的 Expectation 不變（論文設定明顯低於 CE）；本條只補檢定。

#### Key Metrics

- 配對 t、paired bootstrap 與 Welch 的 Holm 校正 p 值；README 判定；各組 Recall@10。

#### Results

| 指標 | BCE（3 seeds） | CE（3 seeds） | 差 | 配對 t p | bootstrap 95% CI | Welch p | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recall@10 | 0.06016 ± 0.00145 | 0.10237 ± 0.00061 | +0.04221 | 8e-212 | +0.0396 到 +0.0449 | 6e-5 | 顯著較好 |
| NDCG@10 | 0.02591 ± 0.00064 | 0.05072 ± 0.00042 | +0.02481 | 4e-242 | +0.0234 到 +0.0262 | 3e-6 | 顯著較好 |

各組 Recall@10（1–4 組／5+ 組）：BCE s43 0.0232／0.0718、s44 0.0182／0.0712；CE s44 0.0291／0.1212；s42、s43 見 2026-11-20 條目。所有 `max_abs_difference` 為 0。

- **Observations:** 兩種檢定都 p < 0.05，差距約為 seed 標準差的 30 倍。

#### Conclusion

H1 依 README 規則成立：只換 CE 顯著較好，且在冷門與熱門組都較好。

## 2026-11-20

### 分組重新評估與逐使用者排名（`STAGE=groups`）

#### Motivation

依目標熱門度拆解整體指標（`Recall/INSIGHT.md` H2），並補出配對檢定需要的逐使用者排名，檢查 H1 的 CE 提升是否只來自熱門商品。

#### Details

- **Variant:** `baseline`
- **Baseline:** 各 run 自己的原紀錄（`best.json`、`test/<set>/metrics.json`）。
- **Change:** 無；只重新評估。
- **Data:** 2014 Beauty ASIN 切分；validation 22,363 筆（1–4 組 4,564、5+ 組 17,718、0 組 81），test 22,363 筆（5,482／16,685／196）。
- **Environment:** SoC，程式 `7209f60`；job ID 記在各 `best-groups.json`（例如 `sasrec-beauty-v1` 1168941、`sasrec-paper-beauty-v1` 1168942）。
- **Command:** `MODEL=sasrec RUN_NAME=<run> SEED=<seed> [TRAIN_ARGS=…] STAGE=groups sbatch --export=ALL --job-name=kuriyo-sasrec scripts/run.sbatch`
- **Artifacts:** 各 run 的 `validation/amazon2014-beauty/best-groups.json`、`best-ranks.json`；`sasrec-beauty-v1` 另有 `test/amazon2014-beauty/groups.json`、`ranks.json`。`sasrec-paper-beauty-s43`、`-s44`、`sasrec-paperce-beauty-s44` 尚未執行。

#### Settings

- 不訓練；以各 run 的 `checkpoints/best.pt` 重新評估，評估設定與原 run 相同（`settings.json`），只多分組與逐使用者排名。分組依該 run 實際訓練的 train。

#### Expectation

在 `Recall/INSIGHT.md` H2 與 TIGER H7 中事先寫下：若內容 ID 有冷門優勢，TIGER 與 item-ID 模型的差距會在 1–4 組與 0 組縮小或反轉。

#### Key Metrics

- 各熱門度組（目標在 train 當目標 0、1–4、5 次以上）的 Recall@10／NDCG@10；整體數字與原紀錄的差（`max_abs_difference`）；逐使用者配對檢定。

#### Results

| run | 設定 | validation 1–4 R@10 | validation 5+ R@10 | test 1–4 R@10 | test 5+ R@10 |
| --- | --- | --- | --- | --- | --- |
| `sasrec-paper-beauty-v1` | 論文設定（BCE） | 0.0184 | 0.0694 | — | — |
| `sasrec-paperce-beauty-v1` | 只換 CE | 0.0289 | 0.1213 | — | — |
| `sasrec-paperce-beauty-s43` | 只換 CE，seed 43 | 0.0294 | 0.1225 | — | — |
| `sasrec-beauty-v1` | CE、hidden 64、2 heads | 0.0302 | 0.1213 | 0.0297 | 0.1042 |

- **Observations:** 所有 run 的 `max_abs_difference` 為 0。0 組全為 0。CE 對 BCE 的提升在 1–4 組 ×1.6、5+ 組 ×1.7。逐使用者配對（seed 42 一對）：CE − BCE Recall@10 +0.043（bootstrap 95% CI +0.040 到 +0.047，paired t p = 5e-128）。`sasrec-beauty-v1` 從 validation 到 test，1–4 組幾乎不變（−2%），5+ 組 −14%。
- **Hypotheses:** CE 的優勢不限於熱門商品（H1 的機制：難負例在各組都存在）。單一 seed 的配對檢定只反映評估樣本雜訊。

#### Conclusion

H1 的 CE 提升遍及各熱門度組；三個 seed 的配對檢定待補 3 個 run 的排名。SASRec 是冷門組的強基準之一（與 FMRec 相當），不是只靠熱門商品。

## 2026-11-19

### 論文設定 vs 只換 CE（H1），各 3 個 seed

#### Motivation

baseline 預設改成論文設定後還沒有完整 run。同時檢驗 [INSIGHT](INSIGHT.md) H1：高於論文的數字主要來自 full-softmax CE，而不是其他差異。

#### Details

- **Variant:** `baseline`（`fa1b700` 起的預設值）
- **Baseline:** 論文設定（預設值，無 `TRAIN_ARGS`）。
- **Change:** 只加 `--loss ce`；hidden 50、1 head、其他設定相同。
- **Data:** 2014 Beauty ASIN 切分。
- **Environment:** SoC，seed 42、43、44；job ID 自動記在各 run。
- **Command:** 論文設定 `MODEL=sasrec RUN_NAME=sasrec-paper-beauty-v1`（seed 42）、`SEED=43 RUN_NAME=sasrec-paper-beauty-s43`、`SEED=44 RUN_NAME=sasrec-paper-beauty-s44`；對照加 `TRAIN_ARGS="--loss ce"`，`RUN_NAME=sasrec-paperce-beauty-v1／-s43／-s44`；皆 `STAGE=train`。
- **Artifacts:** `Recall/SASRec/baseline/runs/<RUN_NAME>/`。

#### Settings

- **資料／前處理：**同下方條目（2014 Beauty 5-core、leave-one-out、歷史上限 20）。
- **模型：**hidden 50、2 層、1 head、dropout 0.5，可學的位置 embedding，歷史上限 20。
- **訓練：**論文設定是 BCE，每個位置一個負例，從使用者 train 序列沒出現過的商品中均勻抽；對照是 full-softmax CE。Adam（0.9、0.98）lr 1e-3、batch 128 個視窗、最多 200 epochs。
- **評估：**每 5 epochs validation，以 NDCG@10 選 `best.pt`，patience 5；只看 validation。

#### Expectation

論文設定的 validation Recall@10 三個 seed 平均落在 0.06–0.08，明顯低於 CE 設定（約 0.10）；只換 CE 的對照平均在 0.09 以上（模型比先前的 64／2 小，可能略低於 0.102）。兩者 Recall@10 平均差至少 0.02，且 seed 層級 Welch t-test p < 0.05。若差距小於 0.01，H1 不成立，高於論文的原因要另找（例如評估協定或資料）。

#### Key Metrics

- 每個 seed 的 best validation Recall@10、NDCG@10 與 best epoch；兩組的平均 ± 標準差、95% CI 與 Welch t-test（README「實驗設計與統計檢定」）。

#### Results

Expectation 寫於 `b112b2c`，在看到結果之前；6 個 job（1166902–1166907，程式 `1d5c37c`）當時已提交。

| 設定 | seed | best epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 論文設定（BCE） | 42 | 110 | 0.03340 | 0.05871 | 0.01721 | 0.02532 |
| | 43 | 155 | 0.03559 | 0.06162 | 0.01821 | 0.02659 |
| | 44 | 115 | 0.03314 | 0.06014 | 0.01710 | 0.02581 |
| 只換 CE | 42 | 85 | 0.06918 | 0.10204 | 0.04015 | 0.05070 |
| | 43 | 90 | 0.06909 | 0.10307 | 0.04017 | 0.05115 |
| | 44 | 65 | 0.06828 | 0.10200 | 0.03942 | 0.05031 |

| 指標 | 論文設定 平均 ± sd［95% CI］ | 只換 CE 平均 ± sd［95% CI］ | 差 | Welch p |
| --- | --- | --- | --- | --- |
| Recall@5 | 0.03404 ± 0.00135［0.03069, 0.03740］ | 0.06885 ± 0.00049［0.06763, 0.07007］ | +0.03480 | 1.2e-4 |
| Recall@10 | 0.06016 ± 0.00145［0.05655, 0.06377］ | 0.10237 ± 0.00061［0.10086, 0.10388］ | +0.04221 | 5.7e-5 |
| NDCG@5 | 0.01751 ± 0.00061［0.01599, 0.01902］ | 0.03992 ± 0.00043［0.03886, 0.04098］ | +0.02241 | 2.7e-6 |
| NDCG@10 | 0.02591 ± 0.00064［0.02432, 0.02750］ | 0.05072 ± 0.00042［0.04967, 0.05176］ | +0.02481 | 3.0e-6 |

- **Observations:** 只換 loss，Recall@10 從 0.060 升到 0.102（+70%），NDCG@10 約加倍；兩組的 seed 標準差都只有 0.0006–0.0015，差距是標準差的約 30 倍。論文設定的 validation 0.060 幾乎等於 TIGER 論文表 3 的 SASRec（0.0605，test）。CE 在 hidden 50／1 head 下（0.1024）與先前 64／2 的 `sasrec-beauty-v1`（0.1023）相同，模型大小不是差距的來源。BCE 收斂較慢（best epoch 110–155，CE 65–90）。
- **Hypotheses:** 使用者層級的配對檢定還無法做（逐使用者排名輸出尚未實作）；但差距遠大於單一 run 的使用者層級標準誤（約 0.002）。

#### Conclusion

H1 成立（Expectation 的門檻：差距 ≥ 0.02 且 Welch p < 0.05，兩者都遠超過）。高於論文的 SASRec 數字來自 full-softmax CE；用論文設定時，Kuriyo 的切分與評估得到和論文相當的數字，也間接說明共用的資料與評估沒有偏高。之後比較 SASRec 時要寫明是哪個 loss。

### Held-out test：`sasrec-beauty-v1`（seed 42）

#### Motivation

`sasrec-beauty-v1` 已訓練完並以 validation 選定 `best.pt`；先取得一次 test 數字，與論文的 test 並列。之後的設定比較仍只看 validation。

#### Details

- **Variant:** `baseline`
- **Baseline:** 同一 run 的 validation（下方 SoC 條目）。
- **Change:** 無；只對已選定的 `best.pt` 評估 test 一次。
- **Data:** 2014 Beauty ASIN 切分的 test（22,363 筆，歷史含 validation 商品），test sha256 `88e1d637…`。
- **Environment:** SoC，程式 `4f6594d`（舊平版面在這次 test 時搬進新版面）；job ID 未記錄（不影響結果；之後重送的 job 1166757 因「test 已跑過」被拒絕）。2026-11-19 之後的 test 會把 Git revision 與 job ID 寫進 `metrics.json`。
- **Command:** `MODEL=sasrec RUN_NAME=sasrec-beauty-v1 STAGE=test sbatch --export=ALL --exclude='xgpe[0-4,6]' --job-name=kuriyo-sasrec scripts/run.sbatch`
- **Artifacts:** SoC 的 `Recall/SASRec/baseline/runs/sasrec-beauty-v1/test/amazon2014-beauty/metrics.json`。

#### Settings

- 與下方 SoC 條目相同（設定記在該 run 的 `settings.json`）。

#### Expectation

事後補記（test 跑完才寫）：test 低於 validation，因為 test 目標的冷門與長尾比例較高（`Recall/INSIGHT.md` Facts）。這個 run 是 `--loss ce --hidden 64 --heads 2` 設定（2026-11-19 以前的預設），不是論文設定。

#### Key Metrics

- test Recall@5/10、NDCG@5/10，與 validation 及論文 test 的差距。

#### Results

| | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- |
| validation（best） | 0.06882 | 0.10227 | 0.04010 | 0.05083 |
| **test** | **0.05388** | **0.08505** | **0.03124** | **0.04133** |
| TIGER 論文表 3 的 SASRec（test） | 0.0387 | 0.0605 | 0.0249 | 0.0318 |

- **Observations:** test Recall@10 比 validation 低 0.0172（約 17%），NDCG@10 低 0.0095。仍比 TIGER 論文表中的 SASRec 高 0.0245（Recall@10），與 `INSIGHT.md` H1（CE loss）的方向一致。
- **Hypotheses:** 單一 seed；使用者層級的標準誤約 0.0019（Recall@10），與其他模型的差距在這個範圍內不能下結論（README「實驗設計與統計檢定」）。

#### Conclusion

CE 設定 seed 42 的 test 是 Recall@10 0.0851、NDCG@10 0.0413。這個 run 不再訓練或調整；之後的 seed 與論文設定另開 run。H1 仍待論文設定的 run 驗證。

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
