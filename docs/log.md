# Work log

Append-only. Newest entries at the bottom.

---

## [2026-11-17] setup | docs/ wiki scaffolded

- Categories: concepts - architecture - subsystems - decisions - reference - operations - history
- Baseline commit: `62e2436`
- Created: `docs/{CLAUDE,index,log}.md`, `raw/`, 4 `.claude/commands/docs-*`,
  and `.claude/scripts/docs-lint.py`

- Started in grow mode：只寫 [[model-family-layout]]（來自 TIGER baseline/ 重整的工作），其他區塊於後續工作時補上。

## [2026-11-17] update | 變體改為自成完整空間

- `Recall/TIGER/` 下所有程式、依賴、tests 搬入 `baseline/`；更新 [[model-family-layout]]。

## [2026-11-18] add | SID RQ-VAE 診斷

- 新增 [[sid-rqvae-training]]：Adagrad accumulator 發散、平均向量捷徑、dead code 三個坑。

## [2026-11-18] add | FMRec baseline

- 新增 [[fmrec]]：`Recall/FMRec/baseline/` 的結構、與作者程式 `1c39b17` 對照的事實，以及 attention mask、評估 RNG、padding 排名等坑。

## [2026-11-18] update | INSIGHT 分層

- 根 `INSIGHT.md` 拆成各層：`Representations/SID/`、`Recall/`、`Ranking/`、`Recall/TIGER/`（`Recall/FMRec/` 已有）。家族根層規則改為可放 `INSIGHT.md`；更新 [[model-family-layout]] 與 [[sid-rqvae-training]] 的來源。

## [2026-11-19] add | SASRec baseline

- 新增 [[sasrec]]：`Recall/SASRec/baseline/` 作為檢查資料與評估的基準，記錄訓練序列重建、位置右對齊與 padding 遮罩的坑；[[model-family-layout]] 改為三個家族。

## [2026-11-19] add | TIGER 訓練與 Adafactor 步長

- 新增 [[tiger-training]]：HF Adafactor `scale_parameter=False` 讓 lr 成為絕對步長，與 T5X 預設不同；新增 `--adafactor-scale-parameter` 與續訓檢查。

## [2026-11-19] update | run.sbatch 的 MODEL 選擇

- [[model-family-layout]]：`scripts/run.sbatch` 新增 `MODEL=sasrec|fmrec`，記錄同時提交多個 job 時用 `STAGE=train` 的坑；[[sid-rqvae-training]] 行號更新。

## [2026-11-19] update | 規則改以 AGENTS.md 為主

- wiki 規則從 `docs/CLAUDE.md` 移到 `docs/AGENTS.md`，四個 `/docs-*` 指令的步驟也併入其 Procedures 一節，任何 agent 都可直接照做；`docs/CLAUDE.md` 只剩 `@AGENTS.md` 匯入，`.claude/commands/` 只剩指向。
- 根 `AGENTS.md` 新增 Agent instruction files 一節；`.agents/`（自動轉換的副本）加入 `.gitignore`。

## [2026-11-19] update | 每個變體一份 Experiment-log

- 新增 `Recall/{TIGER,SASRec,FMRec}/baseline/Experiment-log.md` 與 `Representations/SID/Experiment-log.md`，run 的完整數字從 INSIGHT 搬入；INSIGHT、TODO 依新模板改寫，新增根目錄 `REVIEW.md`。規則寫在根 `AGENTS.md`。
- [[model-family-layout]] 加上變體的 `Experiment-log.md`；[[sid-rqvae-training]]、[[sasrec]] 的數據出處改指 Experiment-log。

## [2026-11-19] update | TIGER 預設改用 T5X 式 Adafactor

- SoC job 1165519 在 20k 的 validation Recall@10 0.0800（舊設定 0.0132），`train.py` 的 `--adafactor-scale-parameter` 與 `run.sbatch` 的 `ADAFACTOR_SCALE_PARAMETER` 預設改為開啟；[[tiger-training]] 補上續訓舊 run 要設 0 的陷阱。行號引用隨 `run.sbatch` 改動更新。

## [2026-11-19] update | 共用 run 目錄與家族層 Experiment-log

- 新增 `core/runs.py`（run 版面、`settings.json`、續訓與 test 檢查、checkpoint、選最佳）、`Recall/core/evaluation.py`，`AsinSplit` 等移入 `Recall/core/data.py`，`experiment.py` 從 `Recall/core/` 移到 `core/`。SASRec、FMRec、TIGER baseline 改用它們；固定 seed 的小 run 在重構前後 loss 與指標逐位相同。
- run 改成 `settings.json`、`train/`、`checkpoints/`、`validation/<set>/`、`test/<set>/`；舊的平版面在續訓或 test 時核對後搬移。新頁 [[run-directory]]。
- Experiment-log 從 `Recall/<family>/baseline/` 移到 `Recall/<family>/`，條目加 Variant 與資料／前處理；[[model-family-layout]]、[[sasrec]]、[[fmrec]]、[[tiger-training]] 更新出處與行號。

## 2026-11-19 — run.sbatch 的 CATEGORY／SEED／TRAIN_ARGS

- [[model-family-layout]]：預設 run 名稱依分類、`SEED`／`TRAIN_ARGS` 傳給 train 與 test；準備腳本改為 `scripts.ensure_amazon_reviews_2014`；`run.sbatch` 行號更新。
- [[sid-rqvae-training]]：`run.sbatch` 行號更新（內容不變）。

## 2026-11-19 — SASRec 預設改成論文設定

- [[sasrec]]：預設 BCE（負例排除看過的商品）、hidden 50／1 head；舊 run 要以 CE 舊值呼叫。
- [[model-family-layout]]：SASRec 預設 RUN_NAME 改為 `sasrec-paper-<分類>-v1`。
- [[run-directory]]：預設值改變時舊 run 的核對、`STAGE=test` 的 checkpoint 檢查。
