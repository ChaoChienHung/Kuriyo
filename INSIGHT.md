# Kuriyo — Research Notes

本文件記錄 Kuriyo 推薦模型專案的研究筆記。`Insights` 記錄從資料或實驗得到的觀察，`Facts` 記錄已確認的資料與實作，`Hypothesis` 記錄待驗證主張。新增實驗結論時，附上資料版本、切分方式、模型設定、評估指標與結果位置。各節依 General、Recall、Ranking、TIGER 分類；TIGER 是 Recall 內的具體模型。

*建立日期：2026-10-25*

## Insights

### General

- `All_Beauty` 的使用者互動極稀疏：631,986 位使用者中，583,553 位只有一筆評論。因此，若要做「每位使用者最後一筆作測試」的離線評估，大部分使用者無法同時出現在訓練與測試中。這是對資料分布的觀察，尚非模型效果結論。
- 2014 `Beauty` 的 9,719 位使用者在 eval 與 test 邊界有相同時間戳。此資料只能用來源行序固定當日內的順序，無法驗證真實互動先後；解讀時間序列結果時須保留這個限制。

### Recall

- 尚無經實驗驗證的 Recall 模型洞見。

### Ranking

- 尚無經實驗驗證的 Ranking 模型洞見。

### TIGER

- 尚無 TIGER 模型實驗結果。

## Facts

### General

- 本 repository 主要用來存放推薦模型，以及相關的資料處理、訓練、評估與實驗紀錄。參見 [README.md](README.md)。
- `data/amazon_reviews/train/`、`data/amazon_reviews/eval/`、`data/amazon_reviews/test/` 已建立；`eval` 指調參用驗證集。2014 `Beauty` 的切分結果已在本地產生，資料檔由 Git 忽略。
- `core/`、`utils/`、`scripts/`、`Recall/` 和 `Ranking/` 已建立。2014 與 2023 下載腳本分別位於 `scripts/download_amazon_reviews_2014.py` 和 `scripts/download_amazon_reviews_2023.py`。
- 已下載並驗證 Amazon Reviews 2023 `All_Beauty` 的 701,528 筆評論及 112,590 筆商品 metadata。評論涵蓋 631,986 位使用者、112,565 個商品；這些商品全部能以 `parent_asin` 對應到 metadata。其中 9,159 位使用者有至少三筆評論，共貢獻 39,427 筆評論。統計由本地 JSONL 逐行計算；來源版本和校驗值記在 `data/amazon_reviews/raw/2023/All_Beauty/manifest.json`。
- 已下載並驗證 Amazon Reviews 2014 `Beauty` 5-core 的 198,502 筆評論與 259,204 筆商品 metadata。評論包含 22,363 位使用者與 12,101 個商品；每位使用者至少有五筆評論，所有評論商品都能以 `asin` 對應到 metadata。來源及校驗值記在 `data/amazon_reviews/raw/2014/Beauty/manifest.json`。

### Recall

- `Recall/` 用於候選商品召回；2014 `Beauty` 的 `examples.jsonl` 已提供使用者歷史與下一件商品目標。現有資料準備指令見 [Recall README](Recall/README.md)。共用 Recall 訓練與評估入口尚未實作。

### Ranking

- `Ranking/` 目前只有 [README](Ranking/README.md)。2014 `Beauty` 的現有序列樣本只有下一件商品正例，尚未建立 Ranking 用候選集、負例或排序標籤，也沒有訓練與評估程式。

### TIGER

- 2014 `Beauty` 的 22,363 位使用者與 12,101 個商品，與 [TIGER 論文](https://arxiv.org/abs/2305.05065) 的 Beauty 資料表相符。
- 2014 `Beauty` 切分方式：依使用者與 `unixReviewTime` 升冪排序；時間戳相同時保留原始檔行序。每位使用者的最後一筆作 test、倒數第二筆作 eval，其餘用於 train；訓練段落展開成前綴預測樣本，歷史最多 20 件。共產生 131,413 筆 train、22,363 筆 eval、22,363 筆 test，以及 12,101 件商品內容資料。規則和來源 SHA-256 記在 `data/amazon_reviews/train/2014/Beauty/split_manifest.json`。
- `Recall/TIGER/` 已有專屬依賴與初始設定；模型訓練與評估程式尚未實作。參見 [TIGER README](Recall/TIGER/README.md)。

## Hypothesis

### General

- **H2 — 時間切分更接近實際使用情境：**依互動時間切分資料，可能比隨機切分更能反映模型對未來互動的預測能力。需比較兩種切分的結果與資料洩漏風險。

### Recall

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的召回模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。

### Ranking

- 尚未提出可測試的 Ranking 假設；需先確定候選集與標籤。

### TIGER

- 尚未提出 TIGER 專屬的可測試假設；可在實作後比較 Semantic ID 與適當基線。

## Discussion

### General

- **候選資料來源：**已為 [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 的 `Beauty` 5-core 與 [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) 的 `All_Beauty` 建立下載流程。2014 年版適合對照 TIGER 論文；2023 年版適合探索較新資料。正式用於訓練前須確認資料品質、規模與使用條件。
- **推薦任務：**需決定要預測評分、排序商品，還是做下一項商品推薦。任務不同，訓練標籤、負樣本與指標也會不同。
- **資料切分：**2014 `Beauty` 已使用每位使用者 leave-one-out 切分，並保留 `data/amazon_reviews/raw/` 原檔。全域時間切分、冷啟動評估與同日時間戳的敏感度分析仍值得另做；不同切分的結果需分開報告。

### Recall

- **首個基線：**可先用熱門商品推薦作為簡單基線，再與使用者—商品互動模型比較。這是待決定的實驗方向，尚非既定方案。

### Ranking

- 排序模型需要明確的候選集來源、負例抽樣及標籤；應先決定是否用 Recall 的輸出作為候選集，再制定離線評估協定。

### TIGER

- [TIGER 論文](https://arxiv.org/abs/2305.05065) 使用 2014 年 Amazon Beauty、Sports and Outdoors、Toys and Games 資料，依使用者時間序列預測下一個商品；保留至少五筆互動的使用者，最後一筆作測試、倒數第二筆作驗證。目前下載的 2023 `All_Beauty` 是不同資料版本，不能直接對照論文指標。

## Problems to solve

### General

1. 確定推薦任務、目標使用情境與成功指標。
2. 選定資料來源與分類，記錄取得方式、版本、欄位及使用條件。
3. 評估同日時間戳排序、冷啟動商品與不同資料切分對結果的影響。

### Recall

1. 建立首個可重現的候選商品召回基線及 Recall@K／NDCG@K 評估。
2. 定義候選集輸出格式，供 Ranking 與其他後續流程共用。

### Ranking

1. 定義候選集來源、負例抽樣、排序標籤與離線評估方式。
2. 在固定候選集上建立 Ranking 基線，避免與 Recall 指標混淆。

### TIGER

1. 實作商品內容向量、RQ-VAE Semantic ID 與 TIGER 序列模型。
2. 記錄模型設定與實驗結果，再檢查和論文資料處理及指標的可比性。
