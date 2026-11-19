# Recall — Research Notes

召回階段的研究筆記：跨家族的比較、基線與共用評估協定。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標與來源。家族自己的發現寫在 `Recall/<family>/INSIGHT.md`，run 紀錄在各家族的 `Recall/<family>/Experiment-log.md`（家族所有變體共用）；總覽見 [根 INSIGHT](../INSIGHT.md)。

*建立：2026-11-18（由根 INSIGHT 分出）；最後更新：2026-11-19。Experiment log：[TIGER](TIGER/Experiment-log.md)、[SASRec](SASRec/Experiment-log.md)、[FMRec](FMRec/Experiment-log.md)。熱門商品基線是一次性的本機計算，數字直接記在本檔 Facts。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- **共用資料與評估沒有問題，TIGER 的差距主要來自 Adafactor 步長：**同一切分、同一個指標函式下，SASRec 與 FMRec 的 validation Recall@10 都約 0.10，舊設定的 TIGER 在 200k 只有 0.034–0.037，換 SID 也沒改善。只把 Adafactor 改成 T5X 式縮放後，TIGER 在 20k 就達 0.080（見下表；`Recall/TIGER/INSIGHT.md` H5）。

## Facts

已確認的資料性質、實作細節與 run 結果。

- `Recall/` 用於候選商品召回；2014 `Beauty` 的 `examples.jsonl` 已提供使用者歷史與下一件商品目標。現有資料準備指令見 [Recall README](../Recall/README.md)。共用 Recall 訓練與評估入口尚未實作。
- 熱門商品基線（2026-11-18，本機）：以每位使用者 train 段落中的所有互動（歷史 + train 目標）計數，固定推薦前 10 名。validation Recall@5/10 = 0.00984 / 0.01704、NDCG@5/10 = 0.00581 / 0.00811；test Recall@5/10 = 0.00729 / 0.01203、NDCG@5/10 = 0.00400 / 0.00552。validation 目標從未出現在該使用者的歷史中（重複率 0）。目前 TIGER validation Recall@10 0.0342 約為熱門基線的 2 倍。
- **冷門與未見過的目標商品（2026-11-19，本機統計 2014 Beauty 切分）：**12,101 件商品中有 59 件從未當過 train 目標，33 件完全沒出現在 train（歷史或目標都沒有）。validation 目標中有 81 筆（0.36%）不曾是 train 目標、51 筆（0.23%）完全沒出現在 train；test 是 196 筆（0.88%）與 138 筆（0.62%）。在 train 當目標少於 5 次的長尾商品，占 validation 目標的 20.8%、test 目標的 25.4%。test 的冷門目標較多，是 test 通常比 validation 低的原因之一。
- **同一切分的 validation 對照（2014 Beauty，22,363 examples，全商品排名）：**

  | 模型 | 設定 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 來源 |
  |---|---|---|---|---|---|---|
  | 熱門商品 | train 段落計數前 10 | 0.00984 | 0.01704 | 0.00581 | 0.00811 | 本機，2026-11-18 |
  | TIGER baseline | 舊 SID（6/26/25）、200k steps | 0.02057 | 0.03421 | 0.01241 | 0.01679 | SoC job 1153307；[log](TIGER/Experiment-log.md) 2026-11-17 |
  | TIGER baseline | 新 SID `tiger-paper-v1`（stable，256/256/256）、200k steps | 0.02182 | 0.03680 | 0.01349 | 0.01835 | SoC job 1163052；[log](TIGER/Experiment-log.md) 2026-11-18 |
  | TIGER baseline | 新 SID、T5X 式 Adafactor（`scale_parameter=True`），20k 最佳（之後下降，100k 0.0546；TIGER INSIGHT H8） | 0.05290 | 0.07999 | 0.03467 | 0.04344 | SoC job 1165519；[log](TIGER/Experiment-log.md) 2026-11-19 |
  | SASRec baseline | 論文設定（BCE、hidden 50、1 head），3 seeds 平均 ± sd | 0.03404 ± 0.00135 | 0.06016 ± 0.00145 | 0.01751 ± 0.00061 | 0.02591 ± 0.00064 | SoC jobs 1166902–04；[log](SASRec/Experiment-log.md) 2026-11-19 |
  | SASRec baseline | 論文設定只換 CE，3 seeds 平均 ± sd | 0.06885 ± 0.00049 | 0.10237 ± 0.00061 | 0.03992 ± 0.00043 | 0.05072 ± 0.00042 | SoC jobs 1166905–07；[log](SASRec/Experiment-log.md) 2026-11-19 |
  | SASRec baseline | CE、hidden 64、2 層，SoC best epoch 85 | 0.06882 | 0.10227 | 0.04010 | 0.05083 | SoC job 1165570；本機同設定 0.09382；[log](SASRec/Experiment-log.md) 2026-11-19 |
  | FMRec baseline | 作者預設，best epoch 160 | 0.07280 | 0.10307 | 0.05058 | 0.06035 | SoC job 1165520；[log](FMRec/Experiment-log.md) 2026-11-19 |

  逐 step／epoch 的數字與 run 證據只記在各變體的 Experiment-log；本表只並列最佳 validation，新增 run 時同時更新 log 與本表。
- **Held-out test（2014 Beauty，22,363 examples，全商品排名，各一次、seed 42）：**

  | 模型 | 設定 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 論文 test（Recall@10／NDCG@10） | 來源 |
  |---|---|---|---|---|---|---|---|
  | 熱門商品 | train 段落計數前 10 | 0.00729 | 0.01203 | 0.00400 | 0.00552 | — | 本機，2026-11-18 |
  | SASRec baseline | CE、hidden 64、2 heads（`sasrec-beauty-v1`） | 0.05388 | 0.08505 | 0.03124 | 0.04133 | 0.0605／0.0318（TIGER 論文表 3） | [log](SASRec/Experiment-log.md) 2026-11-19 |
  | FMRec baseline | 作者預設（`fmrec-beauty-v1`） | 0.05764 | 0.08036 | 0.04017 | 0.04745 | 0.0827／0.0495（FMRec 論文，另一份資料處理） | [log](FMRec/Experiment-log.md) 2026-11-19 |

  test 比 validation 低 17–22%，與 test 的冷門目標較多相符（上一項）。單一 seed，兩模型的差距不足以排序（README「實驗設計與統計檢定」）。TIGER 跑完並定案後才跑 test。

## Limitations

會限制結論範圍的已知弱點。

- 各模型只有單一 seed；SASRec（CE 設定）與 FMRec 各有一次 test，TIGER 只有 validation，T5X Adafactor 的 TIGER 尚未訓練完。
- **目前的評估幾乎不測冷啟動：**catalog 固定，validation 目標只有 0.23% 完全沒出現在 train（見 Facts）。SASRec 與 FMRec 的商品 embedding 隨機初始化、只從互動學習，沒在 train 出現過的商品 embedding 等於沒被訓練過，幾乎不可能被推薦；冷門商品也只有少量更新。現有平均指標看不出這個弱點，因此「item-ID 模型約 0.10」只代表熱門與中段商品上的表現。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/INSIGHT.md` H[n]。

- **H1 — 商品內容有助於推薦：**在相同資料切分下，結合商品 metadata 的召回模型可能優於只看互動紀錄的基線。需先定義冷啟動與整體評估方式，再用實驗驗證。
- **H2 — 以內容產生 ID 的模型在冷門與未見過的商品上優於 item-ID 模型：**TIGER 的 SID 由商品文字的 Sentence-T5 向量產生，相似商品共用前綴 token，新商品不需要互動也有可用的表示；SASRec 與 FMRec 對沒學過的商品只有隨機 embedding。預期：依目標商品在 train 的出現次數分組（0、1–4、5 以上）報告 Recall@10，TIGER 與 item-ID 模型的差距在冷門組縮小或反轉。整體指標由熱門商品主導，看不出這個差異。[TIGER 論文](https://arxiv.org/abs/2305.05065) 另有專門的冷啟動實驗；其設定（例如移除部分商品）與 Kuriyo 的固定 catalog 不同，需另訂協定（根 INSIGHT 與 `TODO.md` 的評估協定延伸）。尚未驗證。

## Discussion

仍在權衡的設計選擇。

- **基線：**熱門商品與 SASRec 基線已完成（見 Facts）。SASRec 已排除切分與指標的問題；FMRec 之後可作為另一個 item-ID 對照。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

1. 定義候選集輸出格式，供 Ranking 與其他後續流程共用。
