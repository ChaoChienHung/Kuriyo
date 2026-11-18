# SASRec — Research Notes

SASRec 家族的研究筆記。`Facts` 是已由程式、資料或 run 確認的內容，`Hypothesis` 是待驗證的主張，兩者分開記錄。操作方式見 [README](README.md) 與 [baseline README](baseline/README.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

格式：`Insights`、`Facts`、`Limitations`、`Hypothesis`、`Discussion`、`Problems to solve`；假設編號只在本檔內有效。

*建立日期：2026-11-19。*

## Insights

- 在 Kuriyo 的 2014 Beauty 切分與共用指標下，SASRec 的 validation Recall@10 達 0.0938，是同切分 TIGER（0.03 左右）的約 3 倍，也高於 TIGER 論文表 3 的 SASRec（0.0605，test）。共用的資料、20 件歷史上限與指標程式能讓成熟模型得到正常甚至偏高的數字，因此 TIGER 的差距不在這些共用部分。

## Facts

- 完整 run `sasrec-beauty-v1`（2026-11-19，本機 Apple MPS，約 23 分鐘）：baseline 預設設定（hidden 64、2 層、2 heads、dropout 0.5、full-softmax CE、Adam lr 1e-3、batch 128 個視窗、seed 42），826,368 參數；131,413 個 train 前綴還原成 23,607 個視窗。每 5 epochs 做一次 validation（22,363 examples，全 catalog 12,101 件排名，不過濾歷史），patience 5，在 epoch 75 early stop。
- validation 曲線：

  | epoch | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
  |---|---|---|---|---|
  | 5 | 0.02039 | 0.03349 | 0.01172 | 0.01593 |
  | 10 | 0.04047 | 0.06122 | 0.02344 | 0.03012 |
  | 20 | 0.05862 | 0.08612 | 0.03323 | 0.04211 |
  | 30 | 0.06310 | 0.09332 | 0.03600 | 0.04572 |
  | **50（best NDCG@10）** | **0.06394** | **0.09382** | **0.03647** | **0.04609** |
  | 60 | 0.06453 | 0.09422 | 0.03653 | 0.04604 |
  | 75（停止） | 0.06202 | 0.09189 | 0.03610 | 0.04573 |

  epoch 30 之後在 Recall@10 0.092–0.094 之間波動；`best.pt` 是 epoch 50。產物在本機 `Recall/SASRec/baseline/runs/sasrec-beauty-v1/`（Git 忽略）。
- `config.json` 記錄的 git revision 是 `97cae0f`，但 SASRec 程式當時尚未 commit（工作目錄含未提交的 `Recall/SASRec/`）；程式內容與之後提交的版本相同。
- 沒有執行 held-out test：模型尚未定案，test 只在定案後跑一次。

## Limitations

- 只跑了一個 seed；validation 不是 test，不能直接和論文的 test 數字比。
- 歷史上限 20（原論文 Beauty 用 50）、預設 CE 而非原論文的 BCE；差異表在 baseline README。

## Hypothesis

- H1：validation 高於 TIGER 論文表 3 的 SASRec，主要來自 full-softmax CE（論文引用的 SASRec 多為 BCE 訓練）。待驗證：`--loss bce` 開新 run 比較。

## Discussion

- SASRec 在此的角色是檢查共用資料與評估，不是追求最佳數字；不必為它調參。

## Problems to solve

1. （選做）`--loss bce` 對照，驗證 H1。
2. 模型定案後對 `best.pt` 跑一次 held-out test。
