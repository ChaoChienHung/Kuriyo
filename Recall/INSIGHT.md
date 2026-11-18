# Recall — Research Notes

召回階段的研究筆記：跨家族的比較、基線與共用評估協定。家族自己的發現寫在 `Recall/<family>/INSIGHT.md`；總覽見 [根 INSIGHT](../INSIGHT.md)。

格式：`Insights`（從資料或實驗得到的觀察）、`Facts`（已確認的資料、實作與 run 結果）、`Limitations`（當前缺點）、`Hypothesis`（待驗證主張）、`Discussion`、`Problems to solve`；事實與假設分開。假設編號只在本檔內有效，跨檔引用時寫檔名加編號（例如 `Recall/TIGER/INSIGHT.md` H3）。

*建立日期：2026-11-18（由根 INSIGHT 分出）。*

## Insights

- 尚無。

## Facts

- `Recall/` 用於候選商品召回；2014 `Beauty` 的 `examples.jsonl` 已提供使用者歷史與下一件商品目標。現有資料準備指令見 [Recall README](../Recall/README.md)。共用 Recall 訓練與評估入口尚未實作。
- 熱門商品基線（2026-11-18，本機）：以每位使用者 train 段落中的所有互動（歷史 + train 目標）計數，固定推薦前 10 名。validation Recall@5/10 = 0.00984 / 0.01704、NDCG@5/10 = 0.00581 / 0.00811；test Recall@5/10 = 0.00729 / 0.01203、NDCG@5/10 = 0.00400 / 0.00552。validation 目標從未出現在該使用者的歷史中（重複率 0）。目前 TIGER validation Recall@10 0.0342 約為熱門基線的 2 倍。
- **同一切分的 validation 對照（2014 Beauty，22,363 examples，全商品排名）：**

  | 模型 | 設定 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 來源 |
  |---|---|---|---|---|---|---|
  | 熱門商品 | train 段落計數前 10 | 0.00984 | 0.01704 | 0.00581 | 0.00811 | 本機，2026-11-18 |
  | TIGER baseline | 舊 SID（6/26/25）、200k steps | 0.02057 | 0.03421 | 0.01241 | 0.01679 | SoC job 1153307；見 [TIGER INSIGHT](TIGER/INSIGHT.md) |
  | FMRec baseline | — | — | — | — | — | 尚未完整訓練；見 [FMRec INSIGHT](FMRec/INSIGHT.md) |

  各家族的詳細數字與 run 證據只記在家族自己的 INSIGHT；本表只做並列，新增 run 時兩邊都要更新同一列。

## Limitations

- 尚無 SASRec 等 item-ID 序列基線；FMRec 已實作但尚未完整訓練，因此還無法判斷 TIGER 的差距有多少來自 SID 或生成式解碼本身。

## Hypothesis

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的召回模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。

## Discussion

- **基線：**熱門商品基線已完成（見 Facts）。下一個非 SID 對照是 FMRec；若 FMRec 在同一切分明顯優於 TIGER，差距較可能來自 SID 或生成解碼，而不是切分或指標。SASRec 仍值得補上作為常見參考點。

## Problems to solve

1. 建立首個可重現的候選商品召回基線及 Recall@K／NDCG@K 評估。
2. 定義候選集輸出格式，供 Ranking 與其他後續流程共用。
