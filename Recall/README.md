# Recall

`Recall/` 放候選商品召回模型。召回階段的目標是根據使用者歷史，從商品目錄找出少量候選商品，供後續評估或 Ranking 使用。[core/](core/README.md) 放不同 Recall 模型共用的指標與後續介面；[TIGER](TIGER/README.md) 是以 SID 生成商品的序列模型，[FMRec](FMRec/README.md) 是以 flow matching 生成商品 embedding 的序列模型，[SASRec](SASRec/README.md) 是以因果 self-attention 預測下一件商品的成熟基準。商品 SID 由獨立的 [Representations/SID](../Representations/SID/README.md) 元件提供。

## 準備 Amazon Reviews 2014 Beauty

從 repository 根目錄執行；資料下載與切分只需要 Python 標準函式庫：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
python3 -m unittest discover -s . -v
```

腳本會將原始資料放在 `data/amazon_reviews/2014/raw/Beauty/`，並產生：

- `data/amazon_reviews/2014/train/raw/Beauty/examples.jsonl`：訓練用的歷史商品序列與下一件商品。
- `data/amazon_reviews/2014/validation/raw/Beauty/examples.jsonl`：每位使用者倒數第二件商品作目標，用於模型選擇。
- `data/amazon_reviews/2014/test/raw/Beauty/examples.jsonl`：每位使用者最後一件商品作目標，用於最終評估。
- `data/amazon_reviews/2014/catalog/Beauty/items.jsonl`：三個 split 共用的商品內容，供內容編碼模型使用。

每筆 `examples.jsonl` 含 `user_id`、`history`、`target_asin`、`target_timestamp`。切分規則、筆數和已知時間戳限制見根目錄 [README](../README.md)。資料檔不納入 Git。

## 模型

- [TIGER](TIGER/README.md)：使用版本化商品 SID 訓練序列模型並計算 validation 指標；首輪完整 SoC 訓練已跑到 200,000 steps，尚無最終 test 結果，未重現論文結果。
- [FMRec](FMRec/README.md)：flow matching 序列推薦，直接使用 item ID embedding 與上方的 ASIN 切分（不需 SID），以共用指標做 full-ranking 評估，可與 TIGER 同切分比較；目前只有測試與 CPU smoke，尚未完整訓練。
- [SASRec](SASRec/README.md)：因果 self-attention 序列推薦，直接使用 item ID embedding 與 ASIN 切分，以共用指標做 full-ranking 評估；用來判斷 TIGER 的差距來自資料／評估還是 TIGER 本身；本機完整 run 的 validation Recall@10 0.0938，見 [INSIGHT.md](INSIGHT.md)。

共用 Recall 訓練入口與候選集輸出格式尚未建立；新增模型時需記錄候選集、評估指標、資料與 SID 版本。每個模型家族的變體各自是一個完整子目錄（例如 `TIGER/baseline/`），包含程式、依賴、測試與 Git 忽略的 runs/；家族根層只放說明（`README.md`）與家族研究筆記（`INSIGHT.md`）。跨家族的比較與基線記在 [INSIGHT.md](INSIGHT.md)。
