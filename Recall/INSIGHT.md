# Recall — Research Notes

召回階段的研究筆記：跨家族的比較、基線與共用評估協定。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標與來源。家族自己的發現寫在 `Recall/<family>/INSIGHT.md`，每個變體的 run 紀錄在 `Recall/<family>/<variant>/Experiment-log.md`；總覽見 [根 INSIGHT](../INSIGHT.md)。

*建立：2026-11-18（由根 INSIGHT 分出）；最後更新：2026-11-19。Experiment log：[TIGER baseline](TIGER/baseline/Experiment-log.md)、[SASRec baseline](SASRec/baseline/Experiment-log.md)、[FMRec baseline](FMRec/baseline/Experiment-log.md)。熱門商品基線是一次性的本機計算，數字直接記在本檔 Facts。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- **共用資料與評估沒有問題，TIGER 的差距在 TIGER 自己的流程：**同一切分、同一個指標函式下，SASRec 與 FMRec 的 validation Recall@10 都在 0.10 左右，TIGER 不論舊 SID（塌縮）或新 SID（256/256/256）在 200k 都只有 0.034–0.037（見下表）。換掉 SID 沒有改善，兩個 item-ID 模型卻都正常，因此下一步應查 TIGER 自己的訓練，而不是資料切分或指標；目前主要嫌疑是 Adafactor 步長（`Recall/TIGER/INSIGHT.md` H5）。

## Facts

已確認的資料性質、實作細節與 run 結果。

- `Recall/` 用於候選商品召回；2014 `Beauty` 的 `examples.jsonl` 已提供使用者歷史與下一件商品目標。現有資料準備指令見 [Recall README](../Recall/README.md)。共用 Recall 訓練與評估入口尚未實作。
- 熱門商品基線（2026-11-18，本機）：以每位使用者 train 段落中的所有互動（歷史 + train 目標）計數，固定推薦前 10 名。validation Recall@5/10 = 0.00984 / 0.01704、NDCG@5/10 = 0.00581 / 0.00811；test Recall@5/10 = 0.00729 / 0.01203、NDCG@5/10 = 0.00400 / 0.00552。validation 目標從未出現在該使用者的歷史中（重複率 0）。目前 TIGER validation Recall@10 0.0342 約為熱門基線的 2 倍。
- **同一切分的 validation 對照（2014 Beauty，22,363 examples，全商品排名）：**

  | 模型 | 設定 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 來源 |
  |---|---|---|---|---|---|---|
  | 熱門商品 | train 段落計數前 10 | 0.00984 | 0.01704 | 0.00581 | 0.00811 | 本機，2026-11-18 |
  | TIGER baseline | 舊 SID（6/26/25）、200k steps | 0.02057 | 0.03421 | 0.01241 | 0.01679 | SoC job 1153307；[log](TIGER/baseline/Experiment-log.md) 2026-11-17 |
  | TIGER baseline | 新 SID `tiger-paper-v1`（stable，256/256/256）、200k steps | 0.02182 | 0.03680 | 0.01349 | 0.01835 | SoC job 1163052；[log](TIGER/baseline/Experiment-log.md) 2026-11-18 |
  | SASRec baseline | CE、hidden 64、2 層，SoC best epoch 85 | 0.06882 | 0.10227 | 0.04010 | 0.05083 | SoC job 1165570；本機同設定 0.09382；[log](SASRec/baseline/Experiment-log.md) 2026-11-19 |
  | FMRec baseline | 作者預設，epoch 80（訓練中） | 0.07119 | 0.10186 | 0.04961 | 0.05954 | SoC job 1165520；[log](FMRec/baseline/Experiment-log.md) 2026-11-19 |

  逐 step／epoch 的數字與 run 證據只記在各變體的 Experiment-log；本表只並列最佳 validation，新增 run 時同時更新 log 與本表。

## Limitations

會限制結論範圍的已知弱點。

- 各模型只有單一 seed、只有 validation；FMRec 尚未訓練完。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/INSIGHT.md` H[n]。

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的召回模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。

## Discussion

仍在權衡的設計選擇。

- **基線：**熱門商品與 SASRec 基線已完成（見 Facts）。SASRec 已排除切分與指標的問題；FMRec 之後可作為另一個 item-ID 對照。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

1. 定義候選集輸出格式，供 Ranking 與其他後續流程共用。
