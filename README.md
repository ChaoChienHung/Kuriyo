# Kuriyo

Kuriyo 這個 repository 主要用來存放**推薦模型**及其相關的資料處理、訓練、評估與實驗紀錄。

專案目前仍在建立初期。資料來源、模型架構和執行方式確定後，會在這裡補上取得資料與重現實驗的步驟。

## 資料目錄

Amazon Reviews 的資料統一放在 `data/amazon_reviews/`：

- `data/amazon_reviews/raw/<年份>/<分類>/`：下載的原始資料及校驗資訊（不納入 Git）
- `data/amazon_reviews/train/`：模型訓練資料
- `data/amazon_reviews/eval/`：調參與選擇模型使用的驗證資料
- `data/amazon_reviews/test/`：最終評估使用的測試資料

資料來源與切分方法確定後，會補上對應說明。大型資料檔不放進 Git。

### Amazon Reviews 2014（TIGER 對應版本）

下載 `Beauty` 的 5-core 評論與商品 metadata：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
```

檔案會存到 `data/amazon_reviews/raw/2014/Beauty/`。腳本也支援 `Sports_and_Outdoors` 和 `Toys_and_Games` 分類。2014 年檔案是 gzip 壓縮的 Python 字典列；解析時應使用 `ast.literal_eval`，不可對下載內容使用 `eval`。資料來源：[McAuley Lab 的 Amazon 2014 資料頁](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。

本次 `Beauty` 下載含 198,502 筆 5-core 評論與 259,204 筆商品 metadata；評論中的 12,101 個商品都能對應到 metadata。

### Amazon Reviews 2023（探索用）

先以 McAuley Lab 的 `All_Beauty` 分類建立資料流程。下載評論及商品 metadata：

```bash
python3 scripts/download_amazon_reviews_2023.py --category All_Beauty
```

檔案會存到 `data/amazon_reviews/raw/2023/All_Beauty/`。兩支下載腳本都會產生記錄來源、筆數及 SHA-256 的 `manifest.json`，只用 Python 標準函式庫。資料來源：[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)。

本次下載的 `All_Beauty` 含 701,528 筆評論與 112,590 筆商品 metadata。下載檔案和本地 manifest 由 `.gitignore` 排除；在其他電腦可用上述指令重新取得。

目前不直接把下載檔案移進 `train/`、`eval/`、`test/`。先保留原始資料；選定推薦任務、互動定義及冷啟動處理方式後，再以固定規則產生切分。每個切分輸出應標明年份、分類與切分版本，避免混用。

若目標是重現 [TIGER](https://arxiv.org/abs/2305.05065) 論文，使用 2014 年資料與對應處理規則；2023 `All_Beauty` 是不同版本，不能直接與論文結果比較。TIGER 將每位使用者的互動依時間排序，保留至少五筆互動的使用者，並把最後兩筆分別用於驗證與測試。

## 程式目錄

- `core/`：推薦模型與共用的核心介面
- `utils/`：資料處理、評估等共用工具
- `scripts/`：執行資料下載、準備、訓練與評估的腳本

`core/` 與 `utils/` 目前尚未加入程式碼。研究筆記與待解決問題記在 [INSIGHT.md](INSIGHT.md)。
