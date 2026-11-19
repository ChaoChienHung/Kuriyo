# SASRec baseline

本目錄是 SASRec 家族的 `baseline` 變體，擁有自己完整的程式、依賴、測試與 runs；不 import 其他變體或其他家族，只用 `Recall/core/`、`core/`。`baseline` 的預設值就是原論文的模型與訓練設定（2026-11-19 起）；其他設定（例如 full-softmax CE）是同一份程式、不同參數的 run，各自記在 `settings.json`。預設值跟論文一致，不代表已重現論文數值。家族說明見 [SASRec 家族 README](../README.md)。

論文：Wang-Cheng Kang、Julian McAuley，**[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)**，ICDM 2018。

目的：在 Kuriyo 的同一份 2014 Beauty leave-one-out 切分、同一個指標函式下，提供一個成熟、容易訓練的參考點，用來判斷 TIGER 與論文的差距來自資料／評估還是 TIGER 本身。[TIGER 論文](https://arxiv.org/abs/2305.05065) 表 3 列出的 SASRec Beauty 結果是 Recall@5 0.0387、NDCG@5 0.0249、Recall@10 0.0605、NDCG@10 0.0318（引自 TIGER 論文，正式引用前需對照原文）。

## 方法

- **輸入**：每個請求取最近 `--max-history`（預設 20）件商品，左側補 0 到固定寬度，最新商品永遠在最後一個位置。可訓練的 item embedding（index 0 是 padding）乘 √hidden，加上依位置學習的 position embedding，再過 dropout；padding 位置歸零。
- **編碼**：`--layers`（預設 2）層 pre-norm Transformer encoder，`--heads` 1、`--hidden` 50（論文設定）、FFN 寬度等於 hidden、ReLU、dropout 0.5，最後一個 LayerNorm。Attention 遮罩是因果遮罩，並且真實位置不 attend padding（`model.py` 的 `attention_allowed`），padding 位置只 attend 自己，避免整列被遮掉而產生 NaN。
- **訓練資料**：從 train 的前綴展開樣本重建每位使用者的完整訓練序列（`data.py` 的 `user_sequences`，逐筆檢查前綴順序），再從序列尾端切成長度不超過 `max_history + 1` 的視窗，每個位置都計算 loss；每個訓練目標剛好出現一次，與 TIGER／FMRec 的訓練目標集合相同。Beauty 共 131,413 個前綴、23,607 個視窗。
- **Loss**：預設 `--loss bce`，原論文作法：每個位置抽一個負例，從使用者 train 序列沒出現過的商品中均勻抽（同原始程式的 `random_neq`；`data.py` 的 `sample_negatives`），正負例各算一次 binary cross-entropy。`--loss ce` 改成在全部 12,101 件商品上做 softmax cross-entropy（共用 item embedding）。
- **評估**：取最後一個位置的輸出與所有 item embedding 做內積，全 catalog 排名，不過濾歷史商品（與 TIGER、FMRec 一致），用共用的 `Recall/core/metrics.py` 算 Recall@5/10、NDCG@5/10。
- **最佳化**：Adam（betas 0.9、0.98）lr 1e-3、weight decay 0、batch 128 個視窗、最多 200 epochs、每 5 epochs 做一次 validation、以 NDCG@10 選 `best.pt`、patience 5。

## 與原論文的差異

| 項目 | 原論文 | 本實作 | 原因 |
| --- | --- | --- | --- |
| Loss | BCE，每位置一個負例 | 相同（預設）；`--loss ce` 可改成 full-softmax | — |
| 歷史長度 | Beauty 用 50 | 20（`--max-history`） | 沿用 Kuriyo 切分（20 件上限來自 TIGER 論文 Appendix C） |
| hidden／heads | 50／1 | 相同（預設） | — |
| 負例 | 排除使用者看過的商品 | 相同：排除使用者 train 序列中的商品 | — |
| 評估 | 1 個正例 + 100 個隨機負例 | 全 catalog 排名 | 與 TIGER、FMRec 和 TIGER 論文的協定一致 |
| 長序列 | 只取最後 50 件 | 切成多個視窗，每個目標都訓練到 | 與 TIGER 的前綴展開使用相同目標 |

## 資料與指標

讀取 `data/amazon_reviews/2014/<train|validation|test>/raw/Beauty/examples.jsonl` 與 `catalog/Beauty/items.jsonl`（產生方式見根目錄 README）。商品 index 依 catalog 檔案順序，1..12,101。`--max-history` 不可超過切分 manifest 記錄的歷史上限。validation 的歷史不含 validation 目標；test 的歷史包含 validation 商品。

## 執行

從 **repository 根目錄**執行，依賴只有根目錄共用的 PyTorch／NumPy：

```bash
python -m pip install -r Recall/SASRec/baseline/requirements.txt
```

完整訓練，論文設定（模型約 64 萬參數；CPU、Apple MPS 或 CUDA 都可以）：

```bash
python -m Recall.SASRec.baseline.train --run-name sasrec-paper-beauty-v1 --device cuda
```

資料放在其他位置時加 `--data-root <path>/data/amazon_reviews`。在 SoC 上用 [`scripts/run.sbatch`](../../../scripts/run.sbatch) 的 `MODEL=sasrec` 提交（資源先依 [SoC 操作說明](../../../skills/nus-soc-slurm/SKILL.md) 核對）。`STAGE=train` 只檢查 Beauty 切分存在就訓練；run 目錄已有 `checkpoints/latest.pt`（或 2026-11-19 以前的 `latest.pt`）時自動續訓，有目錄但沒有 checkpoint 會拒絕；`STAGE=test` 做一次 held-out test：

```bash
MODEL=sasrec STAGE=train sbatch --export=ALL --job-name=kuriyo-sasrec scripts/run.sbatch
```

預設 `RUN_NAME` 是 `sasrec-paper-<分類小寫>-v1`；其他 seed 用 `SEED=43 RUN_NAME=sasrec-paper-beauty-s43`（[README 的統計規則](../../../README.md#實驗設計與統計檢定)）。其他設定是同一份程式加參數，以 `TRAIN_ARGS` 傳入、用自己的 `RUN_NAME`：

| 設定 | `TRAIN_ARGS` | 既有 run |
| --- | --- | --- |
| 論文（預設） | （無） | 尚無 |
| full-softmax CE，hidden 64／2 heads（2026-11-19 以前的預設） | `--loss ce --hidden 64 --heads 2` | `sasrec-beauty-v1`（seed 42）、本機 run |
| 只換 CE（檢驗 INSIGHT H1 的單一因素） | `--loss ce` | 尚無 |

```bash
MODEL=sasrec RUN_NAME=sasrec-ce-beauty-s43 SEED=43 TRAIN_ARGS="--loss ce --hidden 64 --heads 2" STAGE=train sbatch --export=ALL --job-name=kuriyo-sasrec scripts/run.sbatch
```

> ⚠️ `sasrec-beauty-v1` 是用舊預設（CE、64／2）訓練的。對它續訓或跑 test 時必須帶 `TRAIN_ARGS="--loss ce --hidden 64 --heads 2"`，否則設定核對會拒絕（舊 run 的搬移也一樣）。

續訓（從 `checkpoints/latest.pt`，每個 epoch 結束時保存；資料校驗值、seed、batch size、loss、lr 與模型設定必須和 `settings.json` 相同）：

```bash
python -m Recall.SASRec.baseline.train --run-name sasrec-paper-beauty-v1 --device cuda --resume
```

模型定案後，只執行一次 held-out test（拒絕覆寫既有的 `test/<set>/metrics.json`）：

```bash
python -m Recall.SASRec.baseline.train --run-name sasrec-paper-beauty-v1 --device cuda --eval-test-only
```

## 輸出

`Recall/SASRec/baseline/runs/<RUN_NAME>/`（Git 只追蹤 `settings.json`、`events.log`、`validation/`、`test/`）；run 目錄的版面與 `settings.json` 見 [`core/README.md`](../../../core/README.md#run-目錄)。`train/train.jsonl` 每 step 記 loss、learning rate 與 epoch；validation 每 `--eval-every` epochs 寫 `validation/amazon2014-beauty/epoch-<N>.json`。

## 測試

```bash
python -m unittest discover -s Recall/SASRec/baseline/tests -t . -v
```

涵蓋：因果遮罩不看後面的商品、真實位置不 attend padding、padding 位置不影響真實位置的輸出、兩種 loss 有限且梯度傳到每個參數、padding 不進排名、從前綴檔重建序列並拒絕亂序或不連續的資料、視窗讓每個目標剛好訓練一次、評估接到共用指標、訓練／拒絕覆寫／續訓設定不符時拒絕／續訓與不中斷訓練的 loss 相同／test 只能跑一次。這些測試用 CPU 與合成小資料，不代表推薦品質。
