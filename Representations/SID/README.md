# Semantic ID

SID 是商品目錄的版本化表示，不隸屬於某個推薦模型。先用商品內容產生 `semantic_ids.jsonl`，再以 [`scripts/materialize_amazon_sids.py`](../../scripts/materialize_amazon_sids.py) 將同一份對照表套用到 train、validation、test 互動樣本。TIGER 是其中一個使用 SID 的 Recall 模型。

從 repository 根目錄建立環境、安裝依賴並執行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r Representations/SID/requirements.txt
.venv/bin/python -m Representations.SID.scripts.generate_sids \
  --output-dir data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/v1 \
  --encoder sentence-t5 --paper-baseline --device cuda
.venv/bin/python -m scripts.materialize_amazon_sids \
  --year 2014 --category Beauty --method sentence-t5_rqvae \
  --sid-file data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/v1/semantic_ids.jsonl
```

生成器讀 `data/amazon_reviews/2014/catalog/Beauty/items.jsonl` 的 `content_text`，訓練 RQ-VAE 後寫出 `embeddings.npz`、`rqvae.pt`、`semantic_ids.jsonl` 與含來源 SHA、參數、codebook usage 的 `manifest.json`。每個輸出目錄須是新版本，生成器不會覆蓋既有產物。`--run-name` 可改存於 `Representations/SID/runs/<名稱>/` 作試跑；小樣本 SID 不可用來轉換完整 split。

`--paper-baseline` 使用[論文所報告的 RQ-VAE 設定](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)：三層、每層 256 碼、32 維 latent、512/256/128 隱層、β=0.25、Adagrad 0.4、batch 1024、20,000 epochs，並用首批商品的 k-means 初始化 codebook。這是長時間 GPU 工作；輸出 manifest 中的 `codebook_usage_ratio` 應檢查是否接近論文所述的至少 80%。

`sentence-t5_kmeans` 與 `sentence-t5_lsh` 目前只有資料目錄，沒有生成器。目前 2014 Beauty 採固定商品目錄設定：先對 `items.jsonl` 中所有 12,101 件候選商品建立 SID，讓三個 split 共用同一份 ASIN ↔ SID 對照表。這會使用 validation／test 所涉及商品的 metadata，但不使用它們的互動目標來訓練序列模型。若研究目標是訓練時未見商品的歸納能力，須另訂只用訓練期商品擬合 SID 模型、再為新商品指派 SID 的協定，並分開報告結果。
