# TIGER

TIGER（Transformer Index for GEnerative Recommenders）是序列推薦的生成式召回模型：先把商品內容編碼成 Semantic ID（SID），再根據使用者的互動歷史，生成下一件商品的 SID 並查回商品。

論文：Shashank Rajput et al., **[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)**，NeurIPS 2023（[正式論文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)）。本目錄是 Kuriyo 的 PyTorch 實作；目前有可執行的資料、SID、訓練與評估流程，**尚未完成完整 GPU baseline，也尚未重現論文數值**。

下列指令一律從 **Kuriyo repository 根目錄**執行，不是在 `Recall/TIGER/` 裡執行。

## 快速開始：在 SoC 上 pull 後執行

需要 Python 3.10 以上、Slurm GPU allocation、可寫入的資料與模型儲存空間。首次執行需要下載 Amazon 資料、Python 套件及 Sentence-T5 權重。`git pull` 只同步程式碼，這些大型檔案不在 Git 裡。

先依 [SoC 操作說明](../../skills/nus-soc-slurm/SKILL.md) 核對當前 partition、GPU、時限與儲存配額。以下使用 [`scripts/run.sbatch`](../../scripts/run.sbatch) 的範例資源；需要時以 `sbatch` 參數覆蓋。訓練在計算節點執行，不要直接在 login node 啟動 Python 訓練。

```bash
git pull --ff-only

# 同一組實驗的後續階段請沿用這兩個名稱。
export SID_VERSION=tiger-paper-v1
export RUN_NAME=tiger-paper-v1

# 自動建立 .venv、安裝依賴、準備 Beauty，再生成完整商品 SID。
STAGE=sid sbatch --export=ALL scripts/run.sbatch
```

記下回傳的 job ID，以 `squeue -j <job-id>` 查看排隊／執行狀態，並檢查根目錄 `slurm-kuriyo-tiger-<job-id>.out`。工作離開 queue 後，使用 `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS` 確認成功；SID 階段應留下 `semantic_ids.jsonl` 與 `manifest.json`。

**前一階段成功後，才提交下一個階段**，不要一次貼上尚未完成前置工作的指令：

```bash
# SID 成功後：將 train / validation / test 的 ASIN 換成 SID。
STAGE=materialize sbatch --export=ALL scripts/run.sbatch
```

```bash
# materialize 成功後：訓練 TIGER，使用 validation 選 checkpoint。
STAGE=train sbatch --export=ALL scripts/run.sbatch
```

```bash
# 訓練完成、模型定案後：用最佳 checkpoint 評估 held-out test。
STAGE=test sbatch --export=ALL scripts/run.sbatch
```

也可以用 `STAGE=all sbatch --export=ALL scripts/run.sbatch` 在同一個 job 依序完成資料準備、SID、轉換與訓練；它不包含最終 test。完整 SID 訓練預算為 20,000 epochs，TIGER 為 200,000 steps，目前未量測整體 SoC 執行時間，不保證能在單次 allocation 內完成。

| STAGE | 行為與前置條件 |
| --- | --- |
| `prepare` | 下載或驗證 Beauty 原始資料，建立或驗證商品目錄與 ASIN splits |
| `sid` | 先執行 prepare，再生成 SID；已完成的 SID 目錄會重用 |
| `materialize` | 需要完整 SID 與 ASIN splits；轉換三個 split，完整既有輸出會重用 |
| `train` | 需要 materialized splits；既有 run 有 `latest.pt` 時自動續訓 |
| `test` | 需要 run 的 `best.pt`；寫出最終 test 指標，拒絕覆寫已存在的結果 |
| `all`（預設） | 依序 prepare → sid → materialize → train |

所有階段使用相同的 `SID_VERSION`；序列訓練與 test 使用相同的 `RUN_NAME`。新實驗使用新 run 名稱；更換 SID 設定時也使用新 SID 版本。重開 shell 後記得重新 export。若計算節點不能連外，需在允許的節點預先準備依賴、資料與 Hugging Face cache。

## 安裝依賴與手動執行

**SoC Linux x86_64**：根目錄 [`requirements.txt`](../../requirements.txt) 的註解記錄 SoC 安裝方式；指令明確指定 `torch==2.7.1+cu126`，同時安裝 SID 與 TIGER 依賴。`run.sbatch` 已自動套用。手動安裝指令：

```bash
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.7.1+cu126" \
  -r Representations/SID/requirements.txt -r Recall/TIGER/requirements.txt
```

此設定針對叢集 TITAN V（Volta / sm_70）；[CUDA 13 已移除 Volta 函式庫支援](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers)，採用 [PyTorch 官方 CUDA 12.6 build](https://pytorch.org/get-started/previous-versions/)。目前只確認使用者已成功安裝，仍需在 GPU allocation 內驗證運算與完整訓練。本地 CPU／macOS 使用下方一般依賴，不套用此 CUDA 版本覆寫。註解本身不會讓 pip 自動判斷是否位於 SoC，手動安裝時請使用上述完整指令。

若 login node 的 `ulimit -v` 為 `1048576`，代表虛擬記憶體上限 1 GiB，可能導致 `libtorch_cpu.so: failed to map segment`。請在 allocation 內檢查 `ulimit -a` 並測試 Torch，不要將 login node 的載入失敗直接判定為安裝損毀；batch 是否繼承限制亦須確認。

[`requirements.txt`](requirements.txt) 包含 TIGER 訓練需要的依賴：透過根目錄 requirements 引入 NumPy、PyTorch，另列 Hugging Face Transformers（T5 與 Adafactor）。SID 生成另由 [SID requirements](../../Representations/SID/requirements.txt) 管理，安裝 Sentence-Transformers 與 scikit-learn 等依賴。

以下適用本地 GPU 主機或已分配的計算節點；SoC 的 batch script 已自動執行安裝：

```bash
# 尚未建立環境時執行一次。
python3 -m venv .venv
.venv/bin/python -m pip install \
  -r Recall/TIGER/requirements.txt \
  -r Representations/SID/requirements.txt

# 在 GPU allocation 內檢查目前環境。
.venv/bin/python -c 'import torch; from transformers import T5ForConditionalGeneration, Adafactor; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

若已有完整 SID，只需要 TIGER requirements 即可執行序列模型。依賴目前是相容版本範圍，尚非鎖定環境；跑實驗時應另外保存 `pip freeze` 和 GPU 環境資訊。套件安裝成功不代表 CUDA 已能使用，需以上面的檢查確認。

手動執行完整流程（首次執行使用新的輸出版本）：

```bash
export SID_VERSION=tiger-paper-v1
export RUN_NAME=tiger-paper-v1
export SID_DIR="data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/$SID_VERSION"

# 1. 準備原始資料、商品 catalog 與 ASIN splits。
.venv/bin/python -m scripts.prepare_amazon_reviews_2014_beauty

# 2. 對完整商品 catalog 生成 SID。
.venv/bin/python -m Representations.SID.scripts.generate_sids \
  --output-dir "$SID_DIR" --encoder sentence-t5 --paper-baseline --device cuda

# 3. 使用同一份 SID 對照表轉換三個 split。
.venv/bin/python -m scripts.materialize_amazon_sids \
  --year 2014 --category Beauty --method sentence-t5_rqvae \
  --sid-file "$SID_DIR/semantic_ids.jsonl"

# 4. 訓練與 validation。
.venv/bin/python -m Recall.TIGER.scripts.train_paper_baseline \
  --sid-run "$SID_DIR" --sid-method sentence-t5_rqvae \
  --run-name "$RUN_NAME" --device cuda
```

手動 SID 生成器與轉換器會拒絕覆寫既有輸出；重跑時跳過已完成的步驟。序列續訓時，對步驟 4 加上 `--resume`；最終 test 則對同一指令加上 `--eval-test-only`。可以用 `--help` 查看各入口參數。

## 模型如何工作

```text
商品 title / brand / price / categories
  → Sentence-T5 商品向量 → RQ-VAE → ASIN ↔ SID 對照表
使用者的歷史 ASIN
  → 查表轉成 SID 序列 + user token
  → T5 encoder-decoder → 生成下一商品的 SID → 查回 ASIN
```

Kuriyo 的 SID 使用三層量化碼，再附加碰撞消歧 token；相同量化碼的商品依 ASIN 排序決定消歧 token。SID 由獨立的 [`Representations/SID/`](../../Representations/SID/README.md) 負責，其他模型可以共用，無需每個模型各自生成。

目前有兩個序列模型入口，解碼與評估協定不同，結果不能混為同一 baseline：

| 入口 | 用途 | 解碼 |
| --- | --- | --- |
| [`train_paper_baseline.py`](scripts/train_paper_baseline.py) | 依論文已公開設定建立的 HF T5 baseline；`run.sbatch` 使用此入口 | Unconstrained beam search；無效 SID 和重複商品保留排名位置並計為未命中 |
| [`train_sequence.py`](scripts/train_sequence.py) | 小型 PyTorch Transformer，供流程試跑 | 限定有效商品 SID 的 constrained beam search |

正式入口的目前預設值如下，實際定義在 `train_paper_baseline.py`；[`config.py`](config.py) 是小型模型的初始參考設定，不控制正式入口。

| 設定 | 目前實作 |
| --- | --- |
| Encoder / decoder | 各 4 層，d_model=128、d_ff=1024、6 heads、d_kv=64、dropout=0.1 |
| 使用者表示 | SHA-256 映射至 2,000 個 user token buckets |
| 歷史長度 / seed | 最多 20 件商品 / 42 |
| Optimizer / batch / steps | Adafactor / 256 / 200,000 |
| 學習率 | 前 10,000 steps 為 0.01，之後 inverse-square-root decay |
| 保存 / 驗證 | 每 5,000 steps 保存 latest；每 20,000 steps 評估 validation，最後一步也執行 |
| 模型選擇 / beam | 最佳 validation NDCG@10 / beam size 10 |

## 資料與評估協定

目前 batch 流程固定使用 [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 的 **Beauty 5-core** 評論與商品 metadata。2023 All_Beauty 是不同資料集，不能直接拿來與論文對照。下載的 `.json.gz` 會由腳本直接讀取，不需手動解壓縮。

每位使用者依時間排序，同時間戳按來源行序打破平手；排除少於五筆互動的使用者。最後一筆為 test、倒數第二筆為 validation，其餘訓練段落展開為 next-item 前綴樣本，每筆最多保留 20 件歷史商品。來源行序可確保重現，但無法還原同日內未知的真實先後。

本地完整 Beauty 準備結果：12,101 件商品、131,413 train、22,363 validation、22,363 test 樣本。SID 對完整候選商品目錄擬合，使用包含 validation/test 商品在內的內容 metadata；序列模型只對 train 互動訓練。這是固定商品目錄協定，尚非「訓練時完全未見商品」的 cold-start 實驗。

評估計算 Recall@5/10、NDCG@5/10，使用完整商品 SID 詞彙生成候選，不抽樣負例。validation 用於選模型，test 留到模型定案才執行。論文 Beauty 對照值為 Recall@5 **0.0454**、NDCG@5 **0.0321**、Recall@10 **0.0648**、NDCG@10 **0.0384**，見[論文 Table 1](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)；這些不是 Kuriyo 已取得的結果。

## 輸出位置與完成檢查

所有路徑相對 repo 根目錄，資料、權重與 runs 均不提交 Git：

| 產物 | 路徑 |
| --- | --- |
| 原始下載與 manifest | `data/amazon_reviews/2014/raw/Beauty/` |
| 商品 catalog | `data/amazon_reviews/2014/catalog/Beauty/items.jsonl` |
| ASIN 樣本 | `data/amazon_reviews/2014/<train\|validation\|test>/raw/Beauty/examples.jsonl` |
| SID catalog、模型、manifest | `data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/Beauty/<SID_VERSION>/` |
| SID 樣本 | `data/amazon_reviews/2014/<train\|validation\|test>/sids/sentence-t5_rqvae/Beauty/<SID_VERSION>/examples.jsonl` |
| 序列模型實驗 | `Recall/TIGER/runs/<RUN_NAME>/` |

SID 完成後有 `semantic_ids.jsonl`、`manifest.json`、`embeddings.npz`、`rqvae.pt`。manifest 記錄商品來源 SHA、item count 與 codebook usage；應確認完整 12,101 件商品、沒有 `limit_items`，並檢查碼本使用率與碰撞。轉換器會檢查 ASIN 覆蓋是否與 catalog 完全一致，並將來源 SHA 寫入 train SID 目錄的 `split_manifest.json`；訓練入口也會檢查 catalog 與 SID split 的一致性。

序列 run 包含 `config.json`（參數、seed、Git revision、資料 SHA）、`latest.pt`（模型、optimizer 與抽樣狀態）、`best.pt`、`best-metrics.json`、`validation-step-<step>.json`；最終 test 寫入 `test-metrics.json`。`latest.pt` 用於續訓，`best.pt` 用於最終評估。

## 小樣本檢查與測試

在已安裝上述依賴的環境中執行測試，不需要下載完整資料或 GPU：

```bash
.venv/bin/python -m unittest discover -s . -v
```

資料準備完成後，也能用 64 件商品檢查 Sentence-T5 → RQ-VAE 流程。首次需取得 Sentence-T5 權重，輸出使用新的獨立名稱：

```bash
.venv/bin/python -m Representations.SID.scripts.generate_sids \
  --run-name sentence-t5-smoke --encoder sentence-t5 --limit-items 64 \
  --epochs 1 --codebook-size 16 --hidden-dims 64 32 \
  --batch-size 32 --device cpu
```

小樣本 SID 不能用來轉換完整 splits 或訓練正式 baseline。先前 64 商品、1 epoch 的 smoke run 三層碼本使用碼數僅 3、1、1（每層容量 16），顯示 collapse；只能證明流程可執行，不能推論推薦品質。

## 常見問題與目前限制

- **缺少 Beauty catalog：** 更新後的 `STAGE=sid` 會先準備資料；手動流程先執行 `python -m scripts.prepare_amazon_reviews_2014_beauty`。
- **partial download / split 或校驗失敗：** 準備器會停止，保留既有檔案；檢查錯誤指出的路徑與 manifest，不要直接覆寫或刪除原始資料。
- **SID split missing：** 完整 SID 生成後先跑 `STAGE=materialize`，確認後續沿用同一個 `SID_VERSION`。
- **CUDA is unavailable：** 確認執行於 GPU allocation，並檢查該 `.venv` 的 PyTorch/CUDA 與節點驅動。CPU 小樣本可用 `--device cpu`，正式 batch 入口使用 CUDA。
- **run 已存在：** 相同實驗以 `--resume` 接續；新實驗改 `RUN_NAME`。SID 目前沒有中途 checkpoint／續訓，完成前中斷可能需重跑，勿將部分 SID 目錄當成已完成。
- **與論文差距：** 本實作使用 HF T5，並非作者的 T5X；目前約 470 萬參數，低於論文所述約 1,300 萬。user hash、樣本展開、訓練框架及選模協定尚未證實與原實驗完全一致。完整 GPU 結果與鎖定依賴環境仍待補齊。
- **其他 SID 方法：** `sentence-t5_kmeans`、`sentence-t5_lsh` 目前只有預留目錄，尚無生成器。

## 程式導覽

- [`scripts/train_paper_baseline.py`](scripts/train_paper_baseline.py)：T5 建模、訓練、續訓及最終評估。
- [`scripts/train_sequence.py`](scripts/train_sequence.py)、[`sequence.py`](sequence.py)：小型序列模型入口與 SID tokenizer／解碼工具。
- [`Recall/core/`](../core/README.md)：共用 metrics、batching、實驗記錄及基礎 Transformer；測試位於 [`Recall/tests/`](../tests/)。
- [`Representations/SID/`](../../Representations/SID/README.md)：商品向量與 RQ-VAE SID 生成。
- 根目錄 [`INSIGHT.md`](../../INSIGHT.md)、[`TODO.md`](../../TODO.md)：研究事實、假設、限制與待辦事項。
