# Kuriyo

Kuriyo 這個 repository 主要用來存放**推薦模型**及其相關的資料處理、訓練、評估與實驗紀錄。

專案目前仍在建立初期。2014 Amazon Beauty 的資料準備流程已可執行；獨立的 [SID 元件](Representations/SID/README.md)與 [TIGER](Recall/TIGER/README.md) 序列模型已有可執行的試跑流程，尚未重現論文結果。

要執行第一版 TIGER baseline，請從 [TIGER 使用指南](Recall/TIGER/README.md) 開始：包含論文與模型介紹、依賴安裝、SoC pull 後分階段執行、手動指令、續訓、評估及目前限制。

## 資料目錄

Amazon Reviews 的資料依年份存放在 `data/amazon_reviews/2014/` 與 `data/amazon_reviews/2023/`：

```text
data/amazon_reviews/<年份>/
├── raw/<分類>/                       # 下載的原始評論與商品 metadata
├── catalog/<分類>/                   # 三個 split 共用的商品內容目錄
├── artifacts/sids/<方法>/<分類>/<版本>/ # 共用的商品 SID 對照表與生成模型
├── train/
│   ├── raw/<分類>/                   # 切分後、仍用商品 ASIN 的訓練樣本與商品內容
│   └── sids/<方法>/<分類>/<版本>/     # 換成 Semantic ID 的訓練樣本
├── validation/
│   ├── raw/<分類>/
│   └── sids/<方法>/<分類>/<版本>/
└── test/
    ├── raw/<分類>/
    └── sids/<方法>/<分類>/<版本>/
```

`<方法>` 預留 `sentence-t5_rqvae`、`sentence-t5_kmeans` 與 `sentence-t5_lsh`；`<版本>` 區分同一方法的不同 SID run。`artifacts/sids/` 的商品對照表由各模型共用，各 split 的 `sids/` 只保存該版本轉換後的互動樣本。最上層 `raw/` 是下載檔；各 split 內的 `raw/` 是按時間切分後仍保留 ASIN 的樣本。2023 `All_Beauty` 目前只有最上層原始資料，尚未切分或建立 SID。

原始與切分後的大型資料檔不放進 Git；下載和切分規則由腳本與本地 manifest 記錄。

### Amazon Reviews 2014（TIGER 對應版本）

下載 `Beauty` 的 5-core 評論與商品 metadata：

```bash
python3 scripts/download_amazon_reviews_2014.py --category Beauty
```

檔案會存到 `data/amazon_reviews/2014/raw/Beauty/`。腳本也支援 `Sports_and_Outdoors` 和 `Toys_and_Games` 分類。2014 年檔案是 gzip 壓縮的 Python 字典列；解析時應使用 `ast.literal_eval`，不可對下載內容使用 `eval`。資料來源：[McAuley Lab 的 Amazon 2014 資料頁](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。

本次 `Beauty` 下載含 198,502 筆 5-core 評論與 259,204 筆商品 metadata；評論中的 12,101 個商品都能對應到 metadata。

依 [TIGER 論文](https://arxiv.org/abs/2305.05065) 的 leave-one-out 規則產生序列推薦資料：

```bash
python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
python3 -m unittest discover -s . -v
```

準備腳本按 `reviewerID` 分組，以 `unixReviewTime` 升冪排序；同一時間戳以來源檔行序決定先後。少於五筆互動的使用者會排除。每位使用者最後一件商品作 test、倒數第二件作 validation，其餘作 train；每筆樣本的歷史最多保留前 20 件商品。訓練段落會展開為「已有至少一件歷史商品 → 下一件商品」的前綴樣本；這是 Kuriyo 的樣本建構選擇。

每個切分輸出至 `data/amazon_reviews/2014/<train|validation|test>/raw/Beauty/examples.jsonl`，欄位為 `user_id`、`history`、`target_asin`、`target_timestamp`。`2014/catalog/Beauty/items.jsonl` 保存三個 split 共用的 12,101 件商品內容；`split_manifest.json` 記錄來源、商品目錄校驗值與切分規則。

本次 Beauty 輸出為 **131,413 train**、**22,363 validation**、**22,363 test** 筆樣本。2014 年資料的時間戳可能相同：9,719 位使用者的 validation 和 test 事件同時間戳，當日內的實際先後無法由時間戳判定；來源行序使切分可重現，但不代表知道真實先後。

商品 SID 的產生與三個 split 的轉換指令見 [SID README](Representations/SID/README.md)；TIGER 的訓練指令見 [TIGER README](Recall/TIGER/README.md)。`sentence-t5_rqvae` 的完整 Beauty SID 尚待在完整商品目錄上訓練產生；小樣本 smoke test 的 SID 不用作正式 split 資料。

在 SoC 上的完整 Beauty 準備與 TIGER baseline 入口是 [`scripts/run.sbatch`](scripts/run.sbatch)；本地可先用 `python3 -m scripts.prepare_amazon_reviews_2014_beauty` 檢查或準備共用商品目錄與切分。`run.sbatch` 的預設是長時間 GPU 訓練，提交前須查當前叢集資源與儲存規則；細節見 [SoC skill](skills/nus-soc-slurm/SKILL.md)。

若先只生成商品 SID，可從 repo 根目錄執行 `STAGE=sid sbatch --export=ALL scripts/run.sbatch`。這個階段會自動準備 Beauty 原始資料、商品目錄與 ASIN 切分，再生成 SID；完整的既有資料會經檢查後重用。部分下載或不完整的切分會報錯，需先檢查，不會自動覆寫。SID 完成後再以 `STAGE=materialize` 轉換樣本、`STAGE=train` 訓練 TIGER。

### Amazon Reviews 2023（探索用）

先以 McAuley Lab 的 `All_Beauty` 分類建立資料流程。下載評論及商品 metadata：

```bash
python3 scripts/download_amazon_reviews_2023.py --category All_Beauty
```

檔案會存到 `data/amazon_reviews/2023/raw/All_Beauty/`。兩支下載腳本都會產生記錄來源、筆數及 SHA-256 的 `manifest.json`，只用 Python 標準函式庫。資料來源：[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)。

本次下載的 `All_Beauty` 含 701,528 筆評論與 112,590 筆商品 metadata。下載檔案和本地 manifest 由 `.gitignore` 排除；在其他電腦可用上述指令重新取得。

2023 `All_Beauty` 仍只保留原始資料，尚未產生切分。每個切分輸出應標明年份、分類與切分版本，避免混用。

若目標是重現 [TIGER](https://arxiv.org/abs/2305.05065) 論文，使用 2014 年資料與對應處理規則；2023 `All_Beauty` 是不同版本，不能直接與論文結果比較。TIGER 將每位使用者的互動依時間排序，保留至少五筆互動的使用者，並把最後兩筆分別用於驗證與測試。

## 程式目錄

- `core/`：跨鏈路共用的介面；目前尚無實作
- `utils/`：資料處理、評估等共用工具
- `scripts/`：資料下載、準備與相容入口
- [`Representations/`](Representations/README.md)：可供多個模型共用的商品表示；SID 生成器位於 [`Representations/SID/`](Representations/SID/README.md)
- [`Recall/`](Recall/README.md)：候選商品召回模型及資料執行說明
- [`Recall/TIGER/`](Recall/TIGER/README.md)：TIGER 序列模型、訓練與評估
- [`Ranking/`](Ranking/README.md)：候選商品排序模型及資料需求說明
- [`pipelines/`](pipelines/README.md)：預留跨鏈路與端到端推薦流程
- [`skills/`](skills/README.md)：repo 內可版本化的操作技能，目前包含 NUS SoC Slurm

根目錄的 `requirements.txt` 放共用建模依賴；SID 與 TIGER 各有自己的 `requirements.txt`。資料下載與準備只用 Python 標準函式庫；模型環境建議 Python 3.10 以上。各元件的 README 記錄資料指令與實作狀態；研究筆記見 [INSIGHT.md](INSIGHT.md)，待辦事項見 [TODO.md](TODO.md)。

SoC Linux GPU 環境使用根目錄 `requirements-soc.txt`，固定 `torch==2.7.1+cu126`，並引入 SID 與 TIGER 依賴；`scripts/run.sbatch` 自動使用此檔，避免新環境安裝不支援 TITAN V 的 CUDA 13 build。本地 CPU／macOS 仍使用各元件 requirements。這是 GPU 相容性設定，尚非完整依賴 lockfile。
