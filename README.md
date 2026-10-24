# Kuriyo

Kuriyo 這個 repository 主要用來存放**推薦模型**及其相關的資料處理、訓練、評估與實驗紀錄。

專案目前仍在建立初期。資料來源、模型架構和執行方式確定後，會在這裡補上取得資料與重現實驗的步驟。

## 資料目錄

`data/` 目前包含原始資料位置與三個預留的資料集切分：

- `data/raw/`：下載的原始資料及校驗資訊（不納入 Git）
- `data/train/`：模型訓練資料
- `data/validation/`：調整模型與選擇設定時使用的驗證資料
- `data/test/`：最終評估使用的測試資料

資料來源與切分方法確定後，會補上對應說明。大型資料檔不放進 Git。

### Amazon Reviews 2023

先以 McAuley Lab 的 `All_Beauty` 分類建立資料流程。下載評論及商品 metadata：

```bash
python3 scripts/download_amazon_reviews.py --category All_Beauty
```

檔案會存到 `data/raw/amazon_reviews_2023/All_Beauty/`，並產生記錄來源版本、筆數及 SHA-256 的 `manifest.json`。腳本使用 Python 標準函式庫，不需額外安裝套件。資料來源：[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)。

本次下載的 `All_Beauty` 含 701,528 筆評論與 112,590 筆商品 metadata。下載檔案和本地 manifest 由 `.gitignore` 排除；在其他電腦可用上述指令重新取得。

目前不直接把下載檔案移進 `train/`、`validation/`、`test/`。先保留原始資料；選定推薦任務、互動定義及冷啟動處理方式後，再以固定規則產生切分。評估建議先考慮依互動時間切分，並確保測試互動晚於訓練互動。

若目標是重現 [TIGER](https://arxiv.org/abs/2305.05065) 論文，需另用其 2014 年 Amazon Beauty、Sports and Outdoors 或 Toys and Games 資料與對應處理規則；本次下載的 2023 `All_Beauty` 是探索用資料，不能直接與論文結果比較。TIGER 將每位使用者的互動依時間排序，保留至少五筆互動的使用者，並把最後兩筆分別用於驗證與測試。

## 程式目錄

- `core/`：推薦模型與共用的核心介面
- `utils/`：資料處理、評估等共用工具
- `scripts/`：執行資料下載、準備、訓練與評估的腳本

`core/` 與 `utils/` 目前尚未加入程式碼。研究筆記與待解決問題記在 [INSIGHT.md](INSIGHT.md)。
