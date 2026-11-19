# Ranking — Research Notes

排序階段的研究筆記。事實與假設分開，每個結果都註明來源。總覽見 [根 INSIGHT](../INSIGHT.md)。

*建立：2026-11-18（由根 INSIGHT 分出）；最後更新：2026-11-19。Experiment log：尚無（還沒有 Ranking 實驗）。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- 尚無。

## Facts

已確認的資料性質、實作細節與 run 結果。

- `Ranking/` 目前只有 [README](../Ranking/README.md)。2014 `Beauty` 的現有序列樣本只有下一件商品正例，尚未建立 Ranking 用候選集、負例或排序標籤，也沒有訓練與評估程式。

## Limitations

會限制結論範圍的已知弱點。

- 尚未建立候選集、負例、排序標籤及 Ranking 訓練／評估程式，因此目前不能評估排序階段的品質。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Ranking/INSIGHT.md` H[n]。

- 尚無：需先確定候選集與標籤。

## Discussion

仍在權衡的設計選擇。

- 排序模型需要明確的候選集來源、負例抽樣及標籤；應先決定是否用 Recall 的輸出作為候選集，再制定離線評估協定。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

1. 定義候選集來源、負例抽樣、排序標籤與離線評估方式。
2. 在固定候選集上建立 Ranking 基線，避免與 Recall 指標混淆。
