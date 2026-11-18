# Ranking — Research Notes

排序階段的研究筆記。總覽見 [根 INSIGHT](../INSIGHT.md)。

格式：`Insights`（從資料或實驗得到的觀察）、`Facts`（已確認的資料、實作與 run 結果）、`Limitations`（當前缺點）、`Hypothesis`（待驗證主張）、`Discussion`、`Problems to solve`；事實與假設分開。假設編號只在本檔內有效，跨檔引用時寫檔名加編號（例如 `Recall/TIGER/INSIGHT.md` H3）。

*建立日期：2026-11-18（由根 INSIGHT 分出）。*

## Insights

- 尚無。

## Facts

- `Ranking/` 目前只有 [README](../Ranking/README.md)。2014 `Beauty` 的現有序列樣本只有下一件商品正例，尚未建立 Ranking 用候選集、負例或排序標籤，也沒有訓練與評估程式。

## Limitations

- 尚未建立候選集、負例、排序標籤及 Ranking 訓練／評估程式，因此目前不能評估排序階段的品質。

## Hypothesis

- 尚未提出可測試的 Ranking 假設；需先確定候選集與標籤。

## Discussion

- 排序模型需要明確的候選集來源、負例抽樣及標籤；應先決定是否用 Recall 的輸出作為候選集，再制定離線評估協定。

## Problems to solve

1. 定義候選集來源、負例抽樣、排序標籤與離線評估方式。
2. 在固定候選集上建立 Ranking 基線，避免與 Recall 指標混淆。
