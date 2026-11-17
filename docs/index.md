# Kuriyo -- documentation

Kuriyo 是推薦模型研究 repo（Recall、Ranking、SID 表示與端到端流程）。專案總覽與執行方式見根目錄 `README.md`、`AGENTS.md`、`HANDOFF.md`；本 wiki 只記錄這些文件沒寫到的結構說明與踩過的坑，會隨工作逐步擴充。

This directory is a wiki maintained by an LLM: linked markdown pages, nothing outside the
repository. The rules are in [`CLAUDE.md`](CLAUDE.md).

## Start here

1. 根目錄 `README.md` → `AGENTS.md` → `HANDOFF.md`
2. [[model-family-layout]] -- 模型家族與變體（`baseline/` 等）的目錄規則

---

## concepts

（尚無頁面，隨工作補上）

## architecture

- [[model-family-layout]] -- 家族根層放共用程式，變體為同層子目錄，各自有 tests/ 與 runs/

## subsystems

（尚無頁面，隨工作補上）

## decisions

（尚無頁面，隨工作補上）

## reference

（尚無頁面，隨工作補上）

## operations

（尚無頁面，隨工作補上）

## history

（尚無頁面，隨工作補上）

---

## Using this wiki

```
/docs-sync     right after a deploy or release -- reads only what changed since verified_at
/docs-lint     integrity check (orphans, broken links, broken sources, stale, unverified)
/docs-query    ask the docs. If it is not there, the answer says so
/docs-status   commits not yet reflected, stale pages, recent work
```

Every page carries `verified_at`, the commit it was checked against. Change code, run
`/docs-sync`.
