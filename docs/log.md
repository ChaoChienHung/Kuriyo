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
