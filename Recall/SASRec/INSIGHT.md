# SASRec — Research Notes

SASRec 家族的研究筆記。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標，以及來自哪一條 Experiment-log 或哪個 run。操作方式見 [README](README.md) 與 [baseline README](baseline/README.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

*建立：2026-11-19；最後更新：2026-11-19。Experiment log：[baseline](baseline/Experiment-log.md)。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- 在 Kuriyo 的 2014 Beauty 切分與共用指標下，SASRec 的 validation Recall@10 達 0.094–0.102，是同切分 TIGER（0.034–0.037）的約 3 倍，也高於 TIGER 論文表 3 的 SASRec（0.0605，test）。共用的資料、20 件歷史上限與指標程式能讓成熟模型得到正常甚至偏高的數字，因此 TIGER 的差距不在這些共用部分。證據：[baseline Experiment-log](baseline/Experiment-log.md) 2026-11-19。

## Facts

已確認的資料性質、實作細節與 run 結果。

- `sasrec-beauty-v1`（baseline 預設：hidden 64、2 層、full-softmax CE、seed 42，826,368 參數）跑了兩次：SoC CUDA（job 1165570，程式 `7c00c35`）best epoch 85 的 validation Recall@10 0.10227、NDCG@10 0.05083；本機 MPS best epoch 50 的 Recall@10 0.09382、NDCG@10 0.04609。逐 epoch 數字見 [baseline Experiment-log](baseline/Experiment-log.md) 2026-11-19。
- 沒有執行 held-out test：模型尚未定案，test 只在定案後跑一次。

## Limitations

會限制結論範圍的已知弱點。

- 每個環境只跑了一個 seed；SoC 與本機差 0.0085 Recall@10，原因（環境或 seed）未分辨。validation 不是 test，不能直接和論文的 test 數字比。
- 歷史上限 20（原論文 Beauty 用 50）、預設 CE 而非原論文的 BCE；差異表在 baseline README。
- 本機 run 的 `config.json` revision（`97cae0f`）早於 SASRec 程式的 commit，無法以 revision 直接追溯；之後以 SoC run 為主要參考。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/SASRec/INSIGHT.md` H[n]。

- **H1 — 高於論文來自 CE loss：**validation 高於 TIGER 論文表 3 的 SASRec，主要來自 full-softmax CE（論文引用的 SASRec 多為 BCE 訓練）。驗證：`--loss bce` 開新 run 比較。

## Discussion

仍在權衡的設計選擇。

- SASRec 在此的角色是檢查共用資料與評估，不是追求最佳數字；不必為它調參。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

- 尚無阻礙；`--loss bce` 對照與最終 test 已列在 `TODO.md`。
