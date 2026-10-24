# Kuriyo — Research Notes

本文件記錄 Kuriyo 推薦模型專案的研究筆記。請把已確認的事實、從實驗得到的洞見、待驗證的假設及討論分開記錄。新增實驗結論時，附上資料版本、切分方式、模型設定、評估指標與結果位置。

*建立日期：2026-10-25*

## Insights

- `All_Beauty` 的使用者互動極稀疏：631,986 位使用者中，583,553 位只有一筆評論。因此，若要做「每位使用者最後一筆作測試」的離線評估，大部分使用者無法同時出現在訓練與測試中。這是對資料分布的觀察，尚非模型效果結論。
- 目前尚未有 Kuriyo 模型實驗，因此沒有經實驗驗證的模型洞見。

## Facts

- 本 repository 主要用來存放推薦模型，以及相關的資料處理、訓練、評估與實驗紀錄。參見 [README.md](README.md)。
- `data/amazon_reviews/train/`、`data/amazon_reviews/eval/`、`data/amazon_reviews/test/` 已建立，目前只有用來保留目錄的 `.gitkeep`，尚未加入切分資料；`eval` 指調參用驗證集。
- `core/`、`utils/`、`scripts/` 已建立；2014 與 2023 下載腳本分別位於 `scripts/download_amazon_reviews_2014.py` 和 `scripts/download_amazon_reviews_2023.py`。
- 已下載並驗證 Amazon Reviews 2023 `All_Beauty` 的 701,528 筆評論及 112,590 筆商品 metadata。評論涵蓋 631,986 位使用者、112,565 個商品；這些商品全部能以 `parent_asin` 對應到 metadata。其中 9,159 位使用者有至少三筆評論，共貢獻 39,427 筆評論。統計由本地 JSONL 逐行計算；來源版本和校驗值記在 `data/amazon_reviews/raw/2023/All_Beauty/manifest.json`。
- 已下載並驗證 Amazon Reviews 2014 `Beauty` 5-core 的 198,502 筆評論與 259,204 筆商品 metadata。評論包含 22,363 位使用者與 12,101 個商品；每位使用者至少有五筆評論，所有評論商品都能以 `asin` 對應到 metadata。使用者與商品數與 TIGER 論文的 Beauty 資料表相符。來源及校驗值記在 `data/amazon_reviews/raw/2014/Beauty/manifest.json`。
- 目前尚未固定正式訓練資料的處理規則、模型架構或評估協定。

## Hypothesis

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。
- **H2 — 時間切分更接近實際使用情境：**依互動時間切分資料，可能比隨機切分更能反映模型對未來互動的預測能力。需比較兩種切分的結果與資料洩漏風險。

## Discussion

- **候選資料來源：**已為 [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 的 `Beauty` 5-core 與 [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) 的 `All_Beauty` 建立下載流程。2014 年版適合對照 TIGER 論文；2023 年版適合探索較新資料。正式用於訓練前須確認資料品質、規模與使用條件。
- **資料切分：**保留 `data/amazon_reviews/raw/` 原檔，等任務確定後再產生 `train/`、`eval/`、`test/`。應比較全域時間切分、使用者時間切分與冷啟動評估，並明確記錄哪些使用者與商品被排除。
- **與 TIGER 的關係：**[TIGER 論文](https://arxiv.org/abs/2305.05065) 使用 2014 年 Amazon Beauty、Sports and Outdoors、Toys and Games 資料，依使用者時間序列預測下一個商品；保留至少五筆互動的使用者，最後一筆作測試、倒數第二筆作驗證。目前下載的 2023 `All_Beauty` 是不同資料版本，不能直接對照論文指標。
- **推薦任務：**需決定要預測評分、排序商品，還是做下一項商品推薦。任務不同，訓練標籤、負樣本與指標也會不同。
- **首個基線：**可先用熱門商品推薦作為簡單基線，再與使用者—商品互動模型比較。這是待決定的實驗方向，尚非既定方案。

## Problems to solve

1. 確定推薦任務、目標使用情境與成功指標。
2. 選定資料來源與分類，記錄取得方式、版本、欄位及使用條件。
3. 設計可重現的 `train`、`eval`、`test` 切分，避免資料洩漏。
4. 建立資料處理流程與首個可重現的基線。
5. 固定評估協定，記錄模型結果及失敗案例。
