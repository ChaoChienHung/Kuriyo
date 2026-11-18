# Semantic ID

SID 是商品目錄的版本化表示，不隸屬於某個推薦模型。先用商品內容產生 `semantic_ids.jsonl`，再以 [`scripts/materialize_amazon_sids.py`](../../scripts/materialize_amazon_sids.py) 將同一份對照表套用到 train、validation、test 互動樣本。TIGER 是其中一個使用 SID 的 Recall 模型。

從 repository 根目錄建立環境、安裝依賴並執行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r Representations/SID/requirements.txt
.venv/bin/python -m Representations.SID.scripts.generate_sids \
  --output-dir data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/stable-v1 \
  --encoder sentence-t5 --recipe stable --device cuda
.venv/bin/python -m scripts.materialize_amazon_sids \
  --year 2014 --category Beauty --method sentence-t5_rqvae \
  --sid-file data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/stable-v1/semantic_ids.jsonl
```

生成器讀 `data/amazon_reviews/2014/catalog/Beauty/items.jsonl` 的 `content_text`，訓練 RQ-VAE 後寫出 `embeddings.npz`、`rqvae.pt`、`semantic_ids.jsonl` 與含來源 SHA、參數、codebook usage 的 `manifest.json`。每個輸出目錄須是新版本，生成器不會覆蓋既有產物。`--run-name` 可改存於 `Representations/SID/runs/<名稱>/` 作試跑；小樣本 SID 不可用來轉換完整 split。

## 訓練配方（`--recipe`）

三個配方都使用[論文所報告的 RQ-VAE 架構與訓練量](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)：三層、每層 256 碼、32 維 latent、512/256/128 隱層、β=0.25、batch 1024、20,000 epochs，並從隨機 1,024 件商品做 k-means 初始化 codebook。它們只差在怎麼讓訓練穩定：

| `--recipe` | Optimizer | 未使用碼重設 | Embedding 標準化 | 用途 |
| --- | --- | --- | --- | --- |
| `paper-pytorch` | PyTorch Adagrad lr 0.4，accumulator 0 | 否 | 否 | 重現 job 1153307。已知第一個 epoch 就發散、三層塌縮，只作對照。舊旗標 `--paper-baseline` 等同此配方 |
| `paper-tf-adagrad` | Adagrad lr 0.4，accumulator 0.1（TF／Optax 預設） | 是 | 否 | 最貼近論文的穩定版本；本機 60 epochs 碼本全用，但重建只到「輸出平均 embedding」水準 |
| `stable`（`run.sbatch` 預設） | AdamW lr 1e-3 | 是 | 是 | 本機診斷中唯一重建明顯優於平均值、且碼本全用的設定 |

本機診斷數字與 SoC 20,000 epochs 的 `stable` 結果見 [SID INSIGHT](INSIGHT.md)；對 TIGER 指標的影響還沒驗證。標準化時，各維平均值與標準差會一起存進 `embeddings.npz`（`standardize_mean`、`standardize_std`）；開啟重設時，每個 epoch 重設的碼數寫在 `train.jsonl` 的 `dead_codes_reset`，manifest 會記錄配方與總重設數。完成後檢查 manifest 的 `codebook_usage_ratio`（論文稱至少 80%）與碰撞統計。

不指定 `--recipe` 時可以用 `--optimizer`、`--adagrad-initial-accumulator`、`--reset-dead-codes`、`--standardize-embeddings` 組合小型試跑；配方會覆寫這些選項。

改動配方前，先用診斷工具在已有的 `embeddings.npz` 上跑約 60 epochs，確認不塌縮再提交 20,000 epochs 的長時間工作：

```bash
.venv/bin/python -m Representations.SID.scripts.diagnose_rqvae \
  --emb <SID 目錄>/embeddings.npz --recipe stable --epochs 60 --report 20
```

`sentence-t5_kmeans` 與 `sentence-t5_lsh` 目前只有資料目錄，沒有生成器。目前 2014 Beauty 採固定商品目錄設定：先對 `items.jsonl` 中所有 12,101 件候選商品建立 SID，讓三個 split 共用同一份 ASIN ↔ SID 對照表。這會使用 validation／test 所涉及商品的 metadata，但不使用它們的互動目標來訓練序列模型。若研究目標是訓練時未見商品的歸納能力，須另訂只用訓練期商品擬合 SID 模型、再為新商品指派 SID 的協定，並分開報告結果。

## 訓練紀錄

生成器會在輸出目錄保存 `train.jsonl`（每 epoch 的 loss、learning rate、UTC 時間、session ID）與 `events.log`（JSON Lines 執行事件）。這些紀錄不依賴 Slurm；輸出目錄仍由 Git 忽略。輸出目錄會在編碼前建立，因此失敗時可能只有 log；須確認 `semantic_ids.jsonl` 與 `manifest.json` 等必要產物，不能僅以目錄存在判定完成。RQ-VAE 目前沒有中途續訓功能。
