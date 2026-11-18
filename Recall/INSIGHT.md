# Recall — Research Notes

召回階段的研究筆記：跨家族的比較、基線與共用評估協定。家族自己的發現寫在 `Recall/<family>/INSIGHT.md`；總覽見 [根 INSIGHT](../INSIGHT.md)。

格式：`Insights`（從資料或實驗得到的觀察）、`Facts`（已確認的資料、實作與 run 結果）、`Limitations`（當前缺點）、`Hypothesis`（待驗證主張）、`Discussion`、`Problems to solve`；事實與假設分開。假設編號只在本檔內有效，跨檔引用時寫檔名加編號（例如 `Recall/TIGER/INSIGHT.md` H3）。

*建立日期：2026-11-18（由根 INSIGHT 分出）。*

## Insights

- **共用資料與評估沒有問題，TIGER 的差距在 TIGER 自己的流程：**同一切分、同一個指標函式下，SASRec validation Recall@10 0.0938，TIGER 不論舊 SID（塌縮）或新 SID（256/256/256）在 200k 都只有 0.034–0.037（見下表）。換掉 SID 沒有改善，成熟的 item-ID 模型卻正常，因此下一步應查 TIGER 自己的訓練，而不是資料切分或指標；目前主要嫌疑是 Adafactor 步長（`Recall/TIGER/INSIGHT.md` H5）。

## Facts

- `Recall/` 用於候選商品召回；2014 `Beauty` 的 `examples.jsonl` 已提供使用者歷史與下一件商品目標。現有資料準備指令見 [Recall README](../Recall/README.md)。共用 Recall 訓練與評估入口尚未實作。
- 熱門商品基線（2026-11-18，本機）：以每位使用者 train 段落中的所有互動（歷史 + train 目標）計數，固定推薦前 10 名。validation Recall@5/10 = 0.00984 / 0.01704、NDCG@5/10 = 0.00581 / 0.00811；test Recall@5/10 = 0.00729 / 0.01203、NDCG@5/10 = 0.00400 / 0.00552。validation 目標從未出現在該使用者的歷史中（重複率 0）。目前 TIGER validation Recall@10 0.0342 約為熱門基線的 2 倍。
- **同一切分的 validation 對照（2014 Beauty，22,363 examples，全商品排名）：**

  | 模型 | 設定 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 來源 |
  |---|---|---|---|---|---|---|
  | 熱門商品 | train 段落計數前 10 | 0.00984 | 0.01704 | 0.00581 | 0.00811 | 本機，2026-11-18 |
  | TIGER baseline | 舊 SID（6/26/25）、200k steps | 0.02057 | 0.03421 | 0.01241 | 0.01679 | SoC job 1153307；見 [TIGER INSIGHT](TIGER/INSIGHT.md) |
  | TIGER baseline | 新 SID `tiger-paper-v1`（stable，256/256/256）、200k steps | 0.02182 | 0.03680 | 0.01349 | 0.01835 | SoC job 1163052；見 [TIGER INSIGHT](TIGER/INSIGHT.md) |
  | SASRec baseline | CE、hidden 64、2 層，best epoch 50 | 0.06394 | 0.09382 | 0.03647 | 0.04609 | 本機 run `sasrec-beauty-v1`，2026-11-19；見 [SASRec INSIGHT](SASRec/INSIGHT.md) |
  | FMRec baseline | — | — | — | — | — | 尚未完整訓練；見 [FMRec INSIGHT](FMRec/INSIGHT.md) |

  各家族的詳細數字與 run 證據只記在家族自己的 INSIGHT；本表只做並列，新增 run 時兩邊都要更新同一列。

## Limitations

- 各模型只有單一 seed、只有 validation；FMRec 尚未完整訓練。

## Hypothesis

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的召回模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。

## Discussion

- **基線：**熱門商品與 SASRec 基線已完成（見 Facts）。SASRec 已排除切分與指標的問題；FMRec 之後可作為另一個 item-ID 對照。

## Problems to solve

1. 建立首個可重現的候選商品召回基線及 Recall@K／NDCG@K 評估。
2. 定義候選集輸出格式，供 Ranking 與其他後續流程共用。
