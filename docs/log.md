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
