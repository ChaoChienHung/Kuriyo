# Kuriyo

Kuriyo 這個 repository 主要用來存放**推薦模型**及其相關的資料處理、訓練、評估與實驗紀錄。

專案目前仍在建立初期。2014 Amazon Beauty 的資料準備流程已可執行；獨立的 [SID 元件](Representations/SID/README.md)與 [TIGER](Recall/TIGER/README.md) 序列模型已在 SoC 完成首輪完整 Beauty SID 與 200,000-step 訓練；已取得 validation，尚無最終 test 結果，也尚未重現論文結果。

要執行第一版 TIGER baseline，請從 [TIGER 使用指南](Recall/TIGER/README.md) 開始：包含論文與模型介紹、依賴安裝、SoC pull 後分階段執行、手動指令、續訓、評估及目前限制。

## 文件導覽與目前狀態

- [HANDOFF.md](HANDOFF.md)：背景、最新 run 證據、操作限制與接手步驟；內容上限 0.1M tokens。
- [TODO.md](TODO.md)：依優先順序追蹤工作；[INSIGHT.md](INSIGHT.md)：區分觀察、事實、限制與假設。
- [AGENTS.md](AGENTS.md)：協作與修改規則；各元件 README：實際入口與操作方式。

完整 run `tiger-paper-20261117-005232` 的最佳已記錄 validation 位於 step 200,000：Recall@10 **0.03420829**、NDCG@10 **0.01678776**。SID 三層各 256 個碼，實際使用 **6 / 26 / 25**；下一步先調查 SID 與實作設定，不能把 validation 當作論文 test 的直接對照。證據見 [Slurm log](slurm-kuriyo-tiger-1153307.out)。完整 GPU 產物仍在 SoC，Git 只同步程式與小型紀錄。

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

商品 SID 的產生與三個 split 的轉換指令見 [SID README](Representations/SID/README.md)；TIGER 的訓練指令見 [TIGER README](Recall/TIGER/README.md)。`sentence-t5_rqvae` 的完整 Beauty SID 已由 SoC job 1153307 產生，但碼本使用率偏低，仍需診斷；小樣本 smoke test 的 SID 不用作正式 split 資料。

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

- [`core/`](core/README.md)：跨鏈路共用元件，目前包含獨立於 Slurm 的 run logger
- `utils/`：資料處理、評估等共用工具
- [`scripts/`](scripts/README.md)：資料下載、準備、log 匯入與相容入口
- [`Representations/`](Representations/README.md)：可供多個模型共用的商品表示；SID 生成器位於 [`Representations/SID/`](Representations/SID/README.md)
- [`Recall/`](Recall/README.md)：候選商品召回模型及資料執行說明
- [`Recall/TIGER/`](Recall/TIGER/README.md)：TIGER 模型家族；共用 SID tokenizer／解碼，目前執行的 paper-configured 實作在 [`baseline/`](Recall/TIGER/baseline/README.md)
- [`Ranking/`](Ranking/README.md)：候選商品排序模型及資料需求說明
- [`pipelines/`](pipelines/README.md)：預留跨鏈路與端到端推薦流程
- [`skills/`](skills/README.md)：repo 內可版本化的操作技能，目前包含 NUS SoC Slurm

根目錄的 `requirements.txt` 放共用建模依賴；SID 與 TIGER 各有自己的 `requirements.txt`。資料下載與準備只用 Python 標準函式庫；模型環境建議 Python 3.10 以上。各元件的 README 記錄資料指令與實作狀態；研究筆記見 [INSIGHT.md](INSIGHT.md)，待辦事項見 [TODO.md](TODO.md)。

SoC Linux GPU 的安裝指令集中記錄於根目錄 `requirements.txt` 的註解，安裝時明確指定 `torch==2.7.1+cu126` 與 CUDA 12.6 套件來源，並引入 SID 與 TIGER 依賴；`scripts/run.sbatch` 自動套用相同指令，避免新環境安裝不支援 TITAN V 的 CUDA 13 build。本地 CPU／macOS 仍使用各元件 requirements。註解不會改變 pip 行為，手動安裝 SoC 環境時須使用註解中的完整指令。這是 GPU 相容性設定，尚非完整依賴 lockfile。


## 模型家族、變體與測試

相關模型變體集中在同一個家族，例如 `Recall/TIGER/`。跨模型可重用的召回元件放 `Recall/core/`；跨鏈路工具放 `core/`；商品 SID 放 `Representations/SID/`，供不同模型共用。

每個家族目錄的根層放家族共用程式（tokenizer、解碼、資料載入、共用測試）；每個有獨立實作的變體是家族底下的一個子目錄，與其他變體同層，各自包含 `README.md`、入口程式與 `tests/`：

```text
Recall/TIGER/
├── sequence.py, scripts/, tests/   # 家族共用
├── baseline/                       # 目前執行的 paper-configured T5（python -m Recall.TIGER.baseline.train）
│   └── runs/<RUN_NAME>/            # 此變體的 Git 忽略訓練產物
├── <variant-name>/                 # 之後的變體，例如 constrained-decoding/，同樣有自己的 runs/
└── runs/                           # 只放家族層 smoke 工具的產物
```

只有超參數不同時，優先在既有變體加具名設定；確實有不同實作時，才新增變體目錄。名稱表達差異，例如 `baseline`、`constrained-decoding`，避免無法辨識內容的 `variant1`、`v2`。變體 import 家族層與 `Recall/core/` 的共用元件，不每個版本複製一套訓練、評估與 logging。每個變體的訓練產物放在自己目錄下的 `runs/<RUN_NAME>/`，結果不會混到其他變體。`baseline/` 代表比較起點，不代表已重現論文。

測試按責任分層：

- `tests/`：資料準備、SID 共用元件、跨階段流程與 run logging。
- `Recall/tests/`：共用 Transformer、padding、指標等行為契約。
- `Recall/TIGER/tests/`：TIGER 家族共用的 SID 解碼與資料載入。
- `Recall/TIGER/<variant>/tests/`：變體專有測試，例如 `baseline/tests/` 的訓練／續訓／test 整合測試。

修改或新增變體時，同時跑共用元件與模型專有測試；小型測試通過不等於完整資料上的指標已達標。從根目錄執行：

```bash
.venv/bin/python -m unittest discover -s . -v
```

## 獨立訓練紀錄

TIGER paper baseline 和 SID 生成器使用 `core/run_logging.py`，不依賴 Slurm。各自輸出目錄內的 `train.jsonl` 記錄 TIGER 每個 step 或 RQ-VAE 每個 epoch 的 loss、learning rate、UTC 時間、session ID 與本次執行經過秒數；`events.log` 使用 JSON Lines 保存開始、完成及執行區段內的 Python 例外，TIGER 另外記錄續訓、validation、checkpoint 與 test 事件。

TIGER 續訓會追加紀錄；從較早 checkpoint 重跑的 step 可透過 session ID 區分。強制終止、斷電或進入 logger 前的錯誤不保證有失敗事件，因此仍需搭配 Slurm 狀態與產物檢查。SID 失敗後可能留下只有 log 的部分目錄，仍不可當作完成產物，且目前無中途續訓功能。舊的 smoke trainer 尚未接入此 logger。

Slurm `.out` 可提交 Git；`runs/`、SID 產物目錄與 checkpoint 仍被忽略，新增的 run logs 也不會自動同步。要分享結果，需另行匯出選定的小型紀錄。
