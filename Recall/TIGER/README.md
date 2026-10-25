# TIGER

TIGER（Transformer Index for GEnerative Recommenders）用商品內容建立 Semantic ID，再依使用者的商品互動序列預測下一個商品的 Semantic ID。論文：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)。

此目錄存放 TIGER 專屬的設定及後續模型實作。共用資料工具與依賴放在 repository 的 `utils/` 和根目錄 `requirements.txt`。初始參數見 `config.py`；Recall 全體說明見 [Recall README](../README.md)。

## 資料

從 repository 根目錄執行：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
python3 -m unittest discover -s tests -v
```

準備腳本在 `data/amazon_reviews/{train,eval,test}/2014/Beauty/` 寫出 `examples.jsonl`。每行包含 `user_id`、`history`、`target_asin`、`target_timestamp`；訓練資料另有 `items.jsonl`，供商品內容編碼使用。`eval` 是模型選擇用驗證集，`test` 留作最終評估。腳本會寫入 `split_manifest.json` 記錄切分規則與來源校驗值。細節及筆數見根目錄 [README](../../README.md)。

建立模型環境時使用 Python 3.10 以上，並安裝此目錄的依賴（其中會包含根目錄的共用依賴）：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r Recall/TIGER/requirements.txt
```

## 模型階段

1. 從商品的 title、brand、price、category 取得內容向量。
2. 訓練 RQ-VAE，給每件商品建立不重複的 Semantic ID。
3. 將歷史商品 ID 轉成 Semantic ID 序列，訓練 encoder-decoder 預測下一件商品。
4. 用 Recall@5/10、NDCG@5/10 評估完整商品目錄的推薦結果。

目前只有資料準備流程與初始設定可執行；上述模型訓練與評估程式尚待實作，現在沒有 TIGER 模型的執行指令或論文重現分數。
