# TIGER

TIGER（Transformer Index for GEnerative Recommenders）用商品內容建立 Semantic ID，再依使用者的商品互動序列預測下一個商品的 Semantic ID。論文：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)。

此目錄存放 TIGER 的 SID 量化器、序列模型及 SID 產生腳本 [`scripts/generate_sids.py`](scripts/generate_sids.py)。共用資料工具與依賴放在 repository 的 `utils/` 和根目錄 `requirements.txt`。參考參數見 `config.py`；Recall 全體說明見 [Recall README](../README.md)。

## 資料

從 repository 根目錄執行：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
python3 -m unittest discover -s tests -v
```

準備腳本在 `data/amazon_reviews/2014/{train,validation,test}/raw/Beauty/` 寫出 `examples.jsonl`。每行包含 `user_id`、`history`、`target_asin`、`target_timestamp`；訓練資料另有 `items.jsonl`，供商品內容編碼使用。`validation` 是模型選擇用驗證集，`test` 留作最終評估。腳本會寫入 `split_manifest.json` 記錄切分規則與來源校驗值。細節及筆數見根目錄 [README](../../README.md)。

建立模型環境時使用 Python 3.10 以上，並安裝此目錄的依賴（其中會包含根目錄的共用依賴）：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r Recall/TIGER/requirements.txt
```

## SID 與模型階段

1. 從商品的 title、brand、price、categories 建立 `content_text`，再編碼成內容向量。
2. 用 RQ-VAE 的三層殘差量化建立 SID；同一組量化碼發生碰撞時，依 ASIN 排序附加第四個 token，使每件商品有唯一 SID。
3. 將歷史商品的 SID 串成序列，訓練 encoder-decoder Transformer 預測下一件商品的 SID。
4. 在完整商品目錄的有效 SID 上做 constrained beam search，計算 Recall@5/10、NDCG@5/10。

論文使用 Sentence-T5 的 768 維商品向量、RQ-VAE 和 Transformer 序列模型，詳見[原論文](https://arxiv.org/pdf/2305.05065)。此實作的兩階段流程已可跑通，但序列模型是較小的 PyTorch Transformer，評估採限制在商品目錄 SID 的 beam search；尚未重現論文的架構、訓練預算、解碼與結果。`sentence-t5_kmeans` 與 `sentence-t5_lsh` 目前只有資料目錄，尚無產生器。

### Sentence-T5 + RQ-VAE 正式資料路徑

從 repository 根目錄執行。SID 產生器讀 `data/amazon_reviews/2014/train/raw/Beauty/items.jsonl`，用商品的 title、brand、price、categories 組成的 `content_text` 產生 Sentence-T5 向量，再訓練 RQ-VAE。以下 CUDA 指令會把完整商品目錄的模型與 SID 寫進指定的 `train/sids/sentence-t5_rqvae/Beauty/`，目錄已存在時會拒絕覆蓋：

```bash
python3 -m Recall.TIGER.scripts.generate_sids \
  --output-dir data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty \
  --encoder sentence-t5 --device cuda

python3 -m scripts.materialize_amazon_sids \
  --year 2014 --category Beauty --method sentence-t5_rqvae \
  --sid-file data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty/semantic_ids.jsonl

python3 -m scripts.train_tiger_sequence \
  --sid-run data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty \
  --sid-method sentence-t5_rqvae --run-name beauty-sequence --device cuda
```

SID 產生器會寫出 `embeddings.npz`、`rqvae.pt`、`semantic_ids.jsonl`、`manifest.json`。轉換腳本用同一份商品 SID 對照表，把 ASIN 樣本變成各 split 的 `sids/sentence-t5_rqvae/Beauty/examples.jsonl`；訓練目錄另有 `split_manifest.json`。轉換後的樣本包含 `user_id`、`history_sids`、`target_sid`、`target_asin`、`target_timestamp`。這些資料檔與模型權重都由 Git 忽略。序列 run 另存於 `Recall/TIGER/runs/<run-name>/`，名稱不可重複。

### 本地小樣本驗證

若只想確認 Sentence-T5 與 RQ-VAE 都能執行，可把少量商品存到獨立的 smoke run；**少量商品產生的 SID 不能用來轉換完整 train/validation/test**：

```bash
python3 -m Recall.TIGER.scripts.generate_sids \
  --run-name sentence-t5-smoke --encoder sentence-t5 --limit-items 64 \
  --epochs 1 --codebook-size 16 --hidden-dims 64 32 \
  --batch-size 32 --device cpu
```

本機已用 64 件商品完成此 smoke test，但只訓練 1 epoch 時三層 codebook 使用碼數為 3、1、1（容量 16），存在明顯 collapse，不能拿來做推薦品質結論。正式實驗應使用完整商品目錄、足夠訓練步數，並檢查 codebook usage 與碰撞數。預設參數只是可執行起點，不等於論文訓練預算。序列腳本目前只對 `validation` 產生指標，`test` 應在模型定案後才評估。

在 SoC cluster 執行前，先用 `sinfo` 確認可用 partition／GPU、用 `squeue -u "$USER"` 查看自己的工作，再按叢集當前規範設定 `sbatch` 資源與 Python 環境。兩個階段都可使用 `--device cuda`；資料、模型與 log 應放在持久儲存空間。這裡未提交叢集工作。
