# Kuriyo

Kuriyo reproduces sequential recommendation models (generative retrieval with semantic IDs, and item-ID baselines) on Amazon Reviews 2014, under one shared split, one evaluation protocol, and a fixed statistics policy, so that models are compared on equal terms rather than against numbers copied from papers. Named after Kuriyo Urima, the door-to-door salesperson in *Crayon Shin-chan*—just as a salesperson brings the right item to a customer, a recommender finds what users need or want from a large catalog and brings it to them.

## Results (Amazon 2014 Beauty)

Recall@10 / NDCG@10. Kuriyo uses the TIGER protocol: per-user chronological leave-one-out, histories capped at 20 items, ranking over the full catalogue of 12,101 items (no sampled negatives). Validation selects the checkpoint; each test was run once.

| Model | Paper (test) | Kuriyo validation | Kuriyo test | Status |
|---|---|---|---|---|
| [FMRec](Recall/FMRec/README.md) (author defaults) | 0.0827 / 0.0495 | 0.1034 ± 0.0016 / 0.0602 ± 0.0017 (5 seeds) | **0.0804 / 0.0474** (seed 42) | Test within 0.0025 of the paper on all four reported metrics; 5-seed test pending. Beats SASRec with CE loss on NDCG@10 (paired and seed-level tests, 3 shared seeds), ties on Recall@10. The paper used 50-item histories and different preprocessing, so the match is indicative. |
| [SASRec](Recall/SASRec/README.md), paper setting (BCE loss) | 0.0605 / 0.0318¹ | 0.0602 ± 0.0015 / 0.0259 ± 0.0006 (3 seeds) | — | Validation matches the paper; test not run yet. |
| SASRec, cross-entropy loss (hidden 64, 2 heads) | — | 0.1023 / 0.0508 | 0.0851 / 0.0413 | Changing only the loss to full-softmax cross-entropy raises validation Recall@10 from 0.060 to 0.102 (3 seeds each, Welch p = 6e-5). |
| [TIGER](Recall/TIGER/README.md) (T5X-style Adafactor, best at 20k steps) | 0.0648 / 0.0384 | 0.0800 / 0.0434 | 0.0556 / 0.0294 | 14% below the paper on test, and 31% below its own validation; validation declines after 20k steps as training loss keeps falling (overfitting, 1 seed). Under investigation. |
| Most popular items | — | 0.0170 / 0.0081 | 0.0120 / 0.0055 | Floor. |

¹ SASRec as reported in Table 3 of the TIGER paper. Single seed unless stated. Full records: [Recall/INSIGHT.md](Recall/INSIGHT.md) and each family's `Experiment-log.md`.

## Reproduce FMRec

From the repository root, with a CUDA GPU (tested with Python 3.12 and 3.13):

```bash
python -m pip install -r Recall/FMRec/baseline/requirements.txt
python3 -m scripts.ensure_amazon_reviews_2014 --category Beauty   # download, split, checksum
python -m Recall.FMRec.baseline.train --run-name fmrec-repro --device cuda
python -m Recall.FMRec.baseline.train --run-name fmrec-repro --device cuda --eval-test-only
```

On a Slurm cluster the same pipeline is `MODEL=fmrec RUN_NAME=fmrec-repro STAGE=all sbatch --export=ALL scripts/run.sbatch`, then `STAGE=test`. Runs write to `Recall/FMRec/baseline/runs/fmrec-repro/` (layout in [core/README.md](core/README.md)).

## Layout

- `Recall/`: retrieval models, one directory per model family (`TIGER/`, `SASRec/`, `FMRec/`), each with a self-contained `baseline/` variant; shared data loading, full-catalogue evaluation and metrics are in `Recall/core/`.
- `Representations/SID/`: semantic ID generation (Sentence-T5 + RQ-VAE) for generative retrieval.
- `core/`: run-directory lifecycle, provenance, and the statistics used to compare runs (`python -m core.compare`).
- `scripts/`: data download and preparation; `scripts/run.sbatch` runs any stage on Slurm.
- `Ranking/`, `pipelines/`: reserved. The project currently focuses on Recall.

Experiment design, seed counts, and significance tests are fixed in [實驗設計與統計檢定](#實驗設計與統計檢定) below (Chinese). The rest of this README and the research notes are in Traditional Chinese.

Code is released under the [MIT License](LICENSE). The Amazon Reviews data are not redistributed; download them from the [McAuley Lab page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) under its terms.

---

## 簡介（中文）

Kuriyo 這個 repository 主要用來存放**推薦模型**及其相關的資料處理、訓練、評估與實驗紀錄。上方英文結果表是目前的狀態：FMRec 的 test 與論文相當；SASRec 論文設定的 validation 與論文相當；TIGER 的 test 仍比論文低約 14%，正在調查。目前專注在 Recall；`Ranking/` 與 `pipelines/` 保留給之後的重排序與端到端流程。

要執行 TIGER，從 [TIGER baseline 使用指南](Recall/TIGER/baseline/README.md) 開始：包含論文與模型介紹、依賴安裝、SoC pull 後分階段執行、手動指令、續訓、評估及目前限制。

## 文件導覽與目前狀態

- [HANDOFF.md](HANDOFF.md)：背景、最新 run 證據、操作限制與接手步驟；內容上限 0.1M tokens。
- [TODO.md](TODO.md)：未完成的工作（Agent-ready、Owner-led、Ideas）；[REVIEW.md](REVIEW.md)：已完成、等人審查的工作。
- [INSIGHT.md](INSIGHT.md)：研究筆記總覽；各鏈路（Recall、Ranking、SID）與各模型家族有自己的 `INSIGHT.md`，根檔摘要並連結它們。每個模型家族（例如 `Recall/TIGER/`，家族所有變體共用）與 SID 元件各有一份 `Experiment-log.md`，依日期記錄跑了什麼與結果。
- [AGENTS.md](AGENTS.md)：協作與修改規則；各元件 README：實際入口與操作方式。

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

商品 SID 的產生與三個 split 的轉換指令見 [SID README](Representations/SID/README.md)；TIGER 的訓練指令見 [TIGER baseline README](Recall/TIGER/baseline/README.md)。`sentence-t5_rqvae` 的完整 Beauty SID 已由 SoC job 1153307 產生，但碼本使用率偏低，仍需診斷；小樣本 smoke test 的 SID 不用作正式 split 資料。

在 SoC 上的資料準備與訓練入口是 [`scripts/run.sbatch`](scripts/run.sbatch)，以 `CATEGORY` 選分類（預設 `Beauty`）。本地可用 `python3 -m scripts.ensure_amazon_reviews_2014 --category <分類>` 檢查或準備一個分類的商品目錄與切分（下載、切分、校驗，完整的既有資料直接重用）；`--category all` 依序處理論文的三個分類 `Beauty`、`Sports_and_Outdoors`、`Toys_and_Games`。舊指令 `python3 -m scripts.prepare_amazon_reviews_2014_beauty` 等同 `--category Beauty`。三個分類各自切分、各自建立 SID 與 run，不混用。`Sports_and_Outdoors` 已在 SoC 準備（job 1166748，程式 `4f6594d`）：296,337 筆 5-core 評論、35,598 位使用者、18,357 件商品，**189,543 train**、**35,598 validation**、**35,598 test**；使用者與商品數與 TIGER 論文的 Sports 資料表一致。`Toys_and_Games` 已在 SoC 準備（job 1167956，程式 `2b19879`）：167,597 筆 5-core 評論、19,412 位使用者、11,924 件商品，**109,361 train**、**19,412 validation**、**19,412 test**；使用者與商品數與 TIGER 論文的 Toys 資料表一致。`run.sbatch` 的預設是長時間 GPU 訓練，提交前須查當前叢集資源與儲存規則；細節見 [SoC skill](skills/nus-soc-slurm/SKILL.md)。

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
- [`Recall/TIGER/`](Recall/TIGER/README.md)：TIGER 模型家族；每個變體自成完整空間，目前執行的實作在 [`baseline/`](Recall/TIGER/baseline/README.md)
- [`Recall/FMRec/`](Recall/FMRec/README.md)：FMRec（flow matching）模型家族；[`baseline/`](Recall/FMRec/baseline/README.md) 讀 ASIN 切分、不需 SID，入口 `python -m Recall.FMRec.baseline.train`，尚未完整訓練
- [`Recall/SASRec/`](Recall/SASRec/README.md)：SASRec 模型家族，作為檢查資料與評估的成熟基準；[`baseline/`](Recall/SASRec/baseline/README.md) 讀 ASIN 切分，入口 `python -m Recall.SASRec.baseline.train`
- [`Ranking/`](Ranking/README.md)：候選商品排序模型及資料需求說明
- [`pipelines/`](pipelines/README.md)：預留跨鏈路與端到端推薦流程
- [`skills/`](skills/README.md)：repo 內可版本化的操作技能，目前包含 NUS SoC Slurm

根目錄的 `requirements.txt` 放共用建模依賴；SID 與 TIGER 各有自己的 `requirements.txt`。資料下載與準備只用 Python 標準函式庫；模型環境建議 Python 3.10 以上。各元件的 README 記錄資料指令與實作狀態；研究筆記從 [INSIGHT.md](INSIGHT.md) 的各層連結開始，待辦事項見 [TODO.md](TODO.md)。

SoC Linux GPU 的安裝指令集中記錄於根目錄 `requirements.txt` 的註解，安裝時明確指定 `torch==2.7.1+cu126` 與 CUDA 12.6 套件來源，並引入 SID 與 TIGER 依賴；`scripts/run.sbatch` 自動套用相同指令，避免新環境安裝不支援 TITAN V 的 CUDA 13 build。本地 CPU／macOS 仍使用各元件 requirements。註解不會改變 pip 行為，手動安裝 SoC 環境時須使用註解中的完整指令。這是 GPU 相容性設定，尚非完整依賴 lockfile。


## 模型家族、變體與測試

相關模型變體集中在同一個家族，例如 `Recall/TIGER/`。跨模型可重用的召回元件放 `Recall/core/`；跨鏈路工具放 `core/`；商品 SID 放 `Representations/SID/`，供不同模型共用。

每個變體是家族底下的一個子目錄，**自成一個完整的空間**：自己的 `README.md`、`requirements.txt`、模型與訓練程式、入口、`tests/` 及 `runs/<RUN_NAME>/`（只有小型紀錄進 Git，見「獨立訓練紀錄」）。家族根層只放家族 README（說明與變體列表）：

```text
Recall/TIGER/
├── README.md                       # 家族說明與變體列表
├── baseline/                       # 目前執行的實作（python -m Recall.TIGER.baseline.train）
│   ├── README.md, requirements.txt, train.py, sequence.py, ..., tests/
│   └── runs/<RUN_NAME>/
└── <variant-name>/                 # 之後的變體，例如 constrained-decoding/，同樣自成完整空間
```

變體之間不互相 import，修改 baseline 不會悄悄改變其他變體的結果；只可 import `Recall/core/`、`core/`、`Representations/` 等跨模型共用元件。只有設定不同時，用同一個變體加參數跑不同的 run（設定記在 `settings.json`）。**變體一旦有 run 記進 Experiment-log，結果就凍結：**同樣的設定在之後任何 commit 都要得到同樣的 loss 與指標，所以看變體名稱就知道它的行為。不影響結果的修改（logging、run 版面、命令列、錯誤訊息、測試）和預設維持舊行為的新選用參數可以直接改，改前改後各用固定 seed 跑一次確認逐位相同；改預設值、換演算法、或修正會改變結果的 bug，就複製成新的變體目錄，名稱表達差異（例如 `unseen-negatives`、`constrained-decoding`，避免 `variant1`、`v2`），舊變體保留，舊 run 才能照名稱重現。還沒有任何記錄 run 的變體可以自由修改。`baseline/` 的預設值是該家族論文的設定，不代表已重現論文。

測試按責任分層：

- `tests/`：資料準備、SID 共用元件、跨階段流程與 run logging。
- `Recall/tests/`：共用 Transformer、padding、指標等行為契約。
- `Recall/<family>/<variant>/tests/`：變體自己的測試，例如 `TIGER/baseline/tests/` 的 SID 解碼、資料載入與訓練／續訓／test 整合測試，`FMRec/baseline/tests/` 的遮罩、padding invariance、取樣與續訓，`SASRec/baseline/tests/` 的因果遮罩、訓練序列重建與續訓。

修改或新增變體時，同時跑共用元件與該變體的測試；小型測試通過不等於完整資料上的指標已達標。從根目錄執行：

```bash
.venv/bin/python -m unittest discover -s . -v
```

## 實驗設計與統計檢定

比較模型或設定之前先讀這一節；Experiment-log 與 INSIGHT 的結論依這裡的規則下。

**雜訊有兩個來源。**訓練隨機性（初始化、取樣、dropout；換 seed 就會變）與評估樣本（是哪 22,363 位使用者）。單一 run 只能看到後者：Beauty validation 的 Recall@10 約 0.10，使用者層級的標準誤約 √(0.1×0.9／22,363) ≈ 0.002，兩個單一 run 相差 0.005 以內就不能說誰較好。前者要靠多個 seed 才看得到。

**Seed 數：**
- 探索、除錯、找方向：1 個 seed（42），在 Experiment-log 標「單一 seed」，不下優劣結論。
- 要在 INSIGHT 寫「A 比 B 好」：每個設定至少 **3 個 seed（42、43、44）**。
- 定案、與論文並列或對外報告：**5 個 seed（42–46）**。
- 所有被比較的設定用同一組 seed。只有 `run.seed` 不同的 run 以 `-s<seed>` 結尾命名（例如 `sasrec-paper-beauty-s43`）；既有的 `*-v1` run 就是 seed 42。

**報告：**每個指標寫平均 ± 樣本標準差（n 個 seed），以及 t 分佈的 95% 信賴區間（n=3 時半寬 = 4.30·s／√3，n=5 時 2.78·s／√5）。每個資料集分開報告，不跨資料集平均。

**檢定（同一資料集、同一切分）：**
1. **使用者層級配對檢定（主要）：**同一位使用者在兩個模型下的指標配對比較。對每位使用者先平均各 seed 的 NDCG@10（或 Recall@10 是否命中），再做雙尾 paired bootstrap（10,000 次重抽使用者）或 paired t-test；Recall@10 這種 0／1 指標也可用 McNemar。檢定力高，但只反映評估樣本的雜訊。逐使用者排名由評估自動寫出（`best-ranks.json`、`test/<set>/ranks.json`；之前的 run 用 `STAGE=groups` 補），比較用 `python -m core.compare`（見 [core/README.md](core/README.md)），它一併做下面第 2–4 項。
2. **Seed 層級（確認）：**對兩組 seed 的指標做 Welch t-test。n=3 或 5 時檢定力低，只在差距遠大於 seed 標準差時才會顯著，主要用來確認結論不是某個 seed 的運氣。
3. 兩者都 p < 0.05，且平均差距大於兩組 seed 標準差的較大者，才寫「顯著較好」；至少一個檢定 p < 0.05 但條件沒有全部成立時寫「傾向」並說明；兩個檢定都不顯著時，即使差距大於 seed 標準差也不寫優劣（`core.compare` 依此判定）。
4. 同時比較多個模型或資料集時，以 Holm–Bonferroni 校正 p 值。

**Validation 與 test：**模型選擇、早停與上面的檢定都在 validation 上做。設定定案後，每個 seed 的 run 對 test 各跑一次（`STAGE=test`），報告 test 的平均 ± 標準差；不依 test 結果回頭調參。

## 獨立訓練紀錄

TIGER paper baseline、FMRec baseline、SASRec baseline 和 SID 生成器使用 `core/run_logging.py`，不依賴 Slurm。三個 Recall baseline 的 run 目錄版面相同（`settings.json`、`train/`、`checkpoints/`、`validation/<set>/`、`test/<set>/`，見 [core/README.md](core/README.md#run-目錄)）。`train.jsonl` 記錄 TIGER 每個 step 或 RQ-VAE 每個 epoch 的 loss、learning rate、UTC 時間、session ID 與本次執行經過秒數；`events.log` 使用 JSON Lines 保存開始、完成及執行區段內的 Python 例外，TIGER 另外記錄續訓、validation、checkpoint 與 test 事件。

TIGER 續訓會追加紀錄；從較早 checkpoint 重跑的 step 可透過 session ID 區分。強制終止、斷電或進入 logger 前的錯誤不保證有失敗事件，因此仍需搭配 Slurm 狀態與產物檢查。SID 失敗後可能留下只有 log 的部分目錄，仍不可當作完成產物，且目前無中途續訓功能。舊的 smoke trainer 尚未接入此 logger。

Slurm `.out` 被 `.gitignore` 忽略（`/slurm-*.out`）。需要拿 log 補紀錄時（例如 train loss）才 `git add -f` 暫時推上，整理進 Experiment-log、以 `git show <commit>:<檔名>` 連結後就 `git rm`。Recall 的 run 目錄只有 `settings.json`、`events.log`、`validation/`、`test/` 會被 Git 追蹤（`.gitignore` 以白名單列出）；`checkpoints/`、`train/`（每 step 的紀錄，TIGER 可達數十 MB）、2026-11-19 以前平版面的檔案與快取都忽略。在 SoC 上把有 Experiment-log 條目的 run 紀錄 `git add` 後推上來，結果就能在 Git 裡核對；smoke test 的 run 目錄檢查完就刪，不提交。SID 產物目錄仍整個被忽略。
