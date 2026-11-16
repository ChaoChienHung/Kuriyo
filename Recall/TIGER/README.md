# TIGER

TIGER（Transformer Index for GEnerative Recommenders）用商品內容建立 Semantic ID，再依使用者的商品互動序列預測下一個商品的 Semantic ID。論文：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)。

此目錄存放 TIGER 的序列模型、快速試跑入口 [`scripts/train_sequence.py`](scripts/train_sequence.py) 和論文設定入口 [`scripts/train_paper_baseline.py`](scripts/train_paper_baseline.py)。SID 生成器與 RQ-VAE 屬於獨立的 [SID 元件](../../Representations/SID/README.md)，可供其他模型共用；共用評估、資料與 Transformer 元件位於 [`Recall/core/`](../core/README.md)。參考參數見 `config.py`；Recall 全體說明見 [Recall README](../README.md)。

## 資料

從 repository 根目錄執行：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
python3 -m unittest discover -s . -v
```

準備腳本在 `data/amazon_reviews/2014/{train,validation,test}/raw/Beauty/` 寫出 `examples.jsonl`，並在 `data/amazon_reviews/2014/catalog/Beauty/` 寫出共用的 `items.jsonl`。`validation` 是模型選擇用驗證集，`test` 留作最終評估。腳本會寫入 `split_manifest.json` 記錄切分規則與來源校驗值。細節及筆數見根目錄 [README](../../README.md)。

建立模型環境時使用 Python 3.10 以上；完整 SID → TIGER 流程需安裝兩個元件的依賴：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r Recall/TIGER/requirements.txt
.venv/bin/python -m pip install -r Representations/SID/requirements.txt
```

## SID 與模型階段

1. 從商品的 title、brand、price、categories 建立 `content_text`，再編碼成內容向量。
2. 用 RQ-VAE 的三層殘差量化建立 SID；同一組量化碼發生碰撞時，依 ASIN 排序附加第四個 token，使每件商品有唯一 SID。
3. 將歷史商品的 SID 串成序列，訓練 encoder-decoder Transformer 預測下一件商品的 SID。
4. 在完整商品目錄的有效 SID 上做 constrained beam search，計算 Recall@5/10、NDCG@5/10。

論文使用 Sentence-T5 的 768 維商品向量、RQ-VAE 和 Transformer 序列模型，詳見[原論文](https://arxiv.org/pdf/2305.05065)。`train_sequence.py` 是較小的 PyTorch smoke 模型，評估採商品目錄限制的 beam search；`train_paper_baseline.py` 使用 PyTorch/Hugging Face T5、2,000 個 hashed user token、論文所報告的層數與訓練步數，以及不限制商品目錄的 beam search。它仍與原作者的 T5X 實作有差異，尚無完整訓練結果。`sentence-t5_kmeans` 與 `sentence-t5_lsh` 目前只有資料目錄，尚無產生器。

### 使用版本化 SID 訓練 TIGER

從 repository 根目錄執行。先用 [SID README](../../Representations/SID/README.md) 的指令生成商品 SID，並轉換三個 split。`v1` 是此例的 SID 版本，需與生成器的輸出目錄名稱一致：

```bash
.venv/bin/python -m Recall.TIGER.scripts.train_sequence \
  --sid-run data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/v1 \
  --sid-method sentence-t5_rqvae --run-name beauty-sequence --device cuda
```

上述是快速試跑入口。若要對照[論文 Beauty 表格](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)的 Recall@5 0.0454、NDCG@5 0.0321、Recall@10 0.0648、NDCG@10 0.0384，使用完整商品 SID 與下列 T5 入口：

```bash
.venv/bin/python -m Recall.TIGER.scripts.train_paper_baseline \
  --sid-run data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/v1 \
  --sid-method sentence-t5_rqvae --run-name beauty-paper-v1 --device cuda
```

預設 batch 256、200,000 steps、前 10,000 steps 學習率 0.01，之後 inverse-square-root decay；每 5,000 steps 保存可續訓 checkpoint，每 20,000 steps 評估完整 validation 並保留最佳 NDCG@10 checkpoint。`--resume` 從同一 run 的 `latest.pt` 接續。模型定案後才用 `--eval-test-only` 對完整 test 評估一次。這份 PyTorch/Hugging Face 實作依照論文明示的 T5 尺寸設定，但目前模型約 470 萬參數，低於論文所述約 1,300 萬；user hash、訓練框架與最佳模型選擇也未證實與原作者完全一致。因此目標是逐步逼近論文數值，**尚不能宣稱精確復現**。

轉換腳本用同一份商品 SID 對照表，把 ASIN 樣本變成各 split 的 `sids/sentence-t5_rqvae/Beauty/v1/examples.jsonl`；訓練目錄另有 `split_manifest.json`，其中記錄 SID 來源與校驗值。轉換後的樣本包含 `user_id`、`history_sids`、`target_sid`、`target_asin`、`target_timestamp`。這些資料檔與模型權重都由 Git 忽略。序列 run 存於 `Recall/TIGER/runs/<run-name>/`，名稱不可重複。

`STAGE=sid` 會先自動下載或檢查 Beauty 資料並準備共用商品目錄及 ASIN 切分，不需要另外先提交 `STAGE=prepare`。完整資料會重用；部分下載、不完整切分或校驗失敗會停止，保留現場供檢查。

### 本地小樣本驗證

若只想確認 Sentence-T5 與 RQ-VAE 都能執行，可把少量商品存到獨立的 smoke run；**少量商品產生的 SID 不能用來轉換完整 train/validation/test**：

```bash
.venv/bin/python -m Representations.SID.scripts.generate_sids \
  --run-name sentence-t5-smoke --encoder sentence-t5 --limit-items 64 \
  --epochs 1 --codebook-size 16 --hidden-dims 64 32 \
  --batch-size 32 --device cpu
```

本機已用 64 件商品完成此 smoke test，但只訓練 1 epoch 時三層 codebook 使用碼數為 3、1、1（容量 16），存在明顯 collapse，不能拿來做推薦品質結論。正式實驗應使用完整商品目錄、足夠訓練步數，並檢查 codebook usage 與碰撞數。預設參數只是可執行起點，不等於論文訓練預算。序列腳本目前只對 `validation` 產生指標，`test` 應在模型定案後才評估。

在 SoC cluster 上可從 repo 根目錄提交 [`scripts/run.sbatch`](../../scripts/run.sbatch)。預設 `STAGE=all` 依序準備 Beauty 資料、生成 SID、轉換 split、訓練 T5；也可分別用 `STAGE=prepare|sid|materialize|train` 提交，待 validation 選定 checkpoint 後再用 `STAGE=test`。`SID_VERSION` 和 `RUN_NAME` 可選新版本與 run 名稱。pull 只帶程式碼，原始資料與模型依賴不在 Git；首次執行會嘗試在 job 中下載資料及安裝依賴，若叢集計算節點沒有外網，須先在允許的節點準備這兩項。**提交前**依 [repo 內 SoC skill](../../skills/nus-soc-slurm/SKILL.md) 核對當前 partition、GPU、時限、配額及儲存位置，按需以 `sbatch` 參數覆蓋範例資源。這裡未提交叢集工作。
