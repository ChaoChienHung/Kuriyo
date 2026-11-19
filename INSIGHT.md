# Kuriyo — Research Notes

Kuriyo 研究筆記的總覽。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標，以及來自哪一條 Experiment-log 或哪個 run。

筆記分三種檔案：
- **`Experiment-log.md`**：放在每個模型變體裡（`Recall/<family>/<variant>/Experiment-log.md`）或元件旁（`Representations/SID/Experiment-log.md`），依日期記錄跑了什麼、預期、結果與結論。完整數字與 run 證據只寫在這裡。
- **`INSIGHT.md`**：分層存放超出單一 run 的結論。每一層只記自己範圍內的發現並連結 Experiment-log；本檔記跨鏈路的資料與方向，摘要各層結論，最多引用一個關鍵數字。
- 假設編號只在所屬檔案內有效，跨檔引用寫檔名加編號（例如 `Recall/TIGER/INSIGHT.md` H5）。實驗確認或否定假設時，更新該條的狀態並連到實驗，不刪除。

*建立：2026-10-25；2026-11-18 改為分層；最後更新：2026-11-19。交接摘要見 [HANDOFF.md](HANDOFF.md)。*

## 各層筆記

| 檔案 | 範圍 | 目前重點 |
|---|---|---|
| [Representations/SID/INSIGHT.md](Representations/SID/INSIGHT.md) | 與模型無關的商品 SID 產生 | 舊 SID 訓練因 PyTorch Adagrad 第一步發散而塌縮；`stable` 配方在 SoC 20,000 epochs 達 256/256/256，但對 TIGER 只提升 0.0026 Recall@10 |
| [Recall/INSIGHT.md](Recall/INSIGHT.md) | 召回階段、跨家族比較 | 同切分 SASRec、FMRec 都約 0.10：共用資料與評估正常；TIGER 的差距主要來自 Adafactor 步長 |
| [Recall/TIGER/INSIGHT.md](Recall/TIGER/INSIGHT.md) | TIGER 家族 | 舊設定 200k Recall@10 只有 0.034–0.037；改 T5X 式 Adafactor 後 20k 就達 0.080，H5 強烈支持 |
| [Recall/FMRec/INSIGHT.md](Recall/FMRec/INSIGHT.md) | FMRec 家族 | best validation Recall@10 0.1031、NDCG@10 0.0603（目前最高），H2 支持 |
| [Recall/SASRec/INSIGHT.md](Recall/SASRec/INSIGHT.md) | SASRec 家族 | 資料與評估的檢查基準；SoC run validation Recall@10 0.1023 |
| [Ranking/INSIGHT.md](Ranking/INSIGHT.md) | 排序階段 | 尚無候選集、標籤或模型 |

新增一層時，在該目錄建立 `INSIGHT.md` 並加進上表；`pipelines/` 等目前沒有內容的層級，等有發現時再建立。各層的 Experiment-log 由該層 INSIGHT 開頭連結。

## Insights

從資料或實驗得到、會改變後續做法的解讀。各鏈路與模型的解讀在各層檔案。

### Data

- `All_Beauty` 的使用者互動極稀疏：631,986 位使用者中，583,553 位只有一筆評論。因此，若要做「每位使用者最後一筆作測試」的離線評估，大部分使用者無法同時出現在訓練與測試中。這是對資料分布的觀察，尚非模型效果結論。
- 2014 `Beauty` 的 9,719 位使用者在 validation 與 test 邊界有相同時間戳。此資料只能用來源行序固定當日內的順序，無法驗證真實互動先後；解讀時間序列結果時須保留這個限制。

## Facts

已確認的資料性質、實作細節與 run 結果。

### General

- 本 repository 主要用來存放推薦模型，以及相關的資料處理、訓練、評估與實驗紀錄。參見 [README.md](README.md)。
- Amazon Reviews 依年份先分 `2014/`、`2023/`；各年份有來源 `raw/`、共用商品目錄 `catalog/`，以及 `train/`、`validation/`、`test/`。共用商品 SID 目錄位於年份層的 `artifacts/sids/<method>/<category>/<version>/`；各 split 下以 `raw/` 保存 ASIN 樣本，以 `sids/<method>/<category>/<version>/` 保存轉換後的 SID 樣本。預留方法名稱為 `sentence-t5_rqvae`、`sentence-t5_kmeans`、`sentence-t5_lsh`；2023 尚未切分或建立 SID。2014 `Beauty` 的 ASIN 切分結果已在本地產生，資料檔由 Git 忽略。
- `core/`、`utils/`、`scripts/`、`Representations/`、`Recall/`、`Ranking/` 和 `pipelines/` 已建立。2014 與 2023 下載腳本分別位於 `scripts/download_amazon_reviews_2014.py` 和 `scripts/download_amazon_reviews_2023.py`。

### Data

- 已下載並驗證 Amazon Reviews 2023 `All_Beauty` 的 701,528 筆評論及 112,590 筆商品 metadata。評論涵蓋 631,986 位使用者、112,565 個商品；這些商品全部能以 `parent_asin` 對應到 metadata。其中 9,159 位使用者有至少三筆評論，共貢獻 39,427 筆評論。統計由本地 JSONL 逐行計算；來源版本和校驗值記在 `data/amazon_reviews/2023/raw/All_Beauty/manifest.json`。
- 已下載並驗證 Amazon Reviews 2014 `Beauty` 5-core 的 198,502 筆評論與 259,204 筆商品 metadata。評論包含 22,363 位使用者與 12,101 個商品；每位使用者至少有五筆評論，所有評論商品都能以 `asin` 對應到 metadata。來源及校驗值記在 `data/amazon_reviews/2014/raw/Beauty/manifest.json`。
- 2014 `Beauty` 切分方式：依使用者與 `unixReviewTime` 升冪排序；時間戳相同時保留原始檔行序。每位使用者的最後一筆作 test、倒數第二筆作 validation，其餘用於 train；訓練段落展開成前綴預測樣本，歷史最多 20 件。共產生 131,413 筆 train、22,363 筆 validation、22,363 筆 test，以及放在 `2014/catalog/Beauty/` 的 12,101 件商品內容資料。規則和來源 SHA-256 記在 `data/amazon_reviews/2014/train/raw/Beauty/split_manifest.json`。

## Limitations

會限制結論範圍的已知弱點。

### Data

- 2014 `Beauty` 的評論時間戳只精確到日期；9,719 位使用者的 validation 與 test 互動同日。來源行序能讓切分重現，但無法確認當日真實先後，時間序列評估可能受影響。
- 2023 `All_Beauty` 目前只有原始下載檔，尚無可供模型訓練與評估的切分或 SID 資料；2014 與 2023 的結果也不能直接視為同一資料集比較。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `INSIGHT.md` H[n]。

### Data

- **H2 — 時間切分更接近實際使用情境：**依互動時間切分資料，可能比隨機切分更能反映模型對未來互動的預測能力。需比較兩種切分的結果與資料洩漏風險。

## Discussion

仍在權衡的設計選擇。

### General

- **推薦任務：**需決定要預測評分、排序商品，還是做下一項商品推薦。任務不同，訓練標籤、負樣本與指標也會不同。

### Data

- **候選資料來源：**已為 [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 的 `Beauty` 5-core 與 [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) 的 `All_Beauty` 建立下載流程。2014 年版適合對照 TIGER 論文；2023 年版適合探索較新資料。正式用於訓練前須確認資料品質、規模與使用條件。
- **資料切分：**2014 `Beauty` 已使用每位使用者 leave-one-out 切分，並保留 `data/amazon_reviews/<year>/raw/` 原檔。全域時間切分、冷啟動評估與同日時間戳的敏感度分析仍值得另做；不同切分的結果需分開報告。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

### General

1. 確定推薦任務、目標使用情境與成功指標。

### Data

2. 選定資料來源與分類，記錄取得方式、版本、欄位及使用條件。
3. 評估同日時間戳排序、冷啟動商品與不同資料切分對結果的影響。
