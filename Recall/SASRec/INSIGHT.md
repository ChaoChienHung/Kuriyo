# SASRec — Research Notes

SASRec 家族的研究筆記。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標，以及來自哪一條 Experiment-log 或哪個 run。操作方式見 [README](README.md) 與 [baseline README](baseline/README.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

*建立：2026-11-19；最後更新：2026-11-19。Experiment log：[Experiment-log.md](Experiment-log.md)（家族所有變體）。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- **SASRec 的 loss 決定了它是否「高於論文」：**同模型、同資料，只把論文的 BCE（每位置一個隨機負例）換成 full-softmax CE，validation Recall@10 從 0.060 升到 0.102（3 個 seed，差距約為 seed 標準差的 30 倍）。用論文設定時，Kuriyo 的數字與論文相當。比較 SASRec 時一定要寫明 loss；其他論文引用的 SASRec 多半是 BCE 版本。證據：[Experiment-log](Experiment-log.md) 2026-11-19（H1）。
- 在 Kuriyo 的 2014 Beauty 切分與共用指標下，SASRec 的 validation Recall@10 達 0.094–0.102，是同切分 TIGER（0.034–0.037）的約 3 倍，也高於 TIGER 論文表 3 的 SASRec（0.0605，test）；`sasrec-beauty-v1` 的 test Recall@10 是 0.0851（[Experiment-log](Experiment-log.md) 2026-11-19 Held-out test），同樣高於論文。共用的資料、20 件歷史上限與指標程式能讓成熟模型得到正常甚至偏高的數字，因此 TIGER 的差距不在這些共用部分。證據：[baseline Experiment-log](Experiment-log.md) 2026-11-19。

## Facts

已確認的資料性質、實作細節與 run 結果。

- `sasrec-beauty-v1`（baseline 預設：hidden 64、2 層、full-softmax CE、seed 42，826,368 參數）跑了兩次：SoC CUDA（job 1165570，程式 `7c00c35`）best epoch 85 的 validation Recall@10 0.10227、NDCG@10 0.05083；本機 MPS best epoch 50 的 Recall@10 0.09382、NDCG@10 0.04609。逐 epoch 數字見 [baseline Experiment-log](Experiment-log.md) 2026-11-19。
- 沒有執行 held-out test：模型尚未定案，test 只在定案後跑一次。

## Limitations

會限制結論範圍的已知弱點。

- 每個環境只跑了一個 seed；SoC 與本機差 0.0085 Recall@10，原因（環境或 seed）未分辨。validation 不是 test，不能直接和論文的 test 數字比。
- 歷史上限 20（原論文 Beauty 用 50）、預設 CE 而非原論文的 BCE；差異表在 baseline README。
- 商品 embedding 隨機初始化、只從互動學習，對沒在 train 出現過的商品無能為力，冷門商品也學得少；目前的評估幾乎不測冷啟動（`Recall/INSIGHT.md` Limitations、H2）。
- 本機 run 的 `config.json` revision（`97cae0f`）早於 SASRec 程式的 commit，無法以 revision 直接追溯；之後以 SoC run 為主要參考。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/SASRec/INSIGHT.md` H[n]。

- **H1 — 高於論文來自 CE loss（成立）：**validation 高於 TIGER 論文表 3 的 SASRec（0.0605），主要來自 full-softmax CE。原始 SASRec 用 BCE，每個正例只配一個隨機負例；TIGER 表中的 SASRec 數字很可能沿用這類設定的早期文獻（尚未對照原文確認）。機制：隨機抽到的負例大多是冷門、與使用者無關的商品，模型很快就能把它們分開，loss 與梯度隨之變小，學到的界線只區分「目標」與「隨便一件商品」；評估卻是在全部 12,101 件中排名，真正的對手是和目標相似、分數也高的商品。full-softmax CE 每一步都和所有商品比較，每個負例的梯度與它的 softmax 機率成正比，所以分數高、難分的負例被推得最多，直接對應全商品排名。文獻也觀察到 CE 訓練的 SASRec 明顯強於 BCE 版本（Klenitskiy & Vasilev, RecSys 2023；Petrov & Macdonald 的 gSASRec, RecSys 2023 指出負例抽樣造成的過度自信）。驗證：2026-11-19 起 baseline 預設就是論文的 BCE 設定（hidden 50／1 head，負例排除看過的商品）；對照只加 `--loss ce`，兩邊各 3 個 seed。預期論文設定的 validation Recall@10 明顯低於 0.10，接近論文的 0.06。目前 0.10 的數字來自 `--loss ce --hidden 64 --heads 2` 設定。結果（各 3 個 seed）：論文設定 Recall@10 0.0602 ± 0.0015，只換 CE 0.1024 ± 0.0006，差 0.042，Welch p = 6e-5；論文設定的數字與論文的 0.0605 相當。[Experiment-log](Experiment-log.md) 2026-11-19。

## Discussion

仍在權衡的設計選擇。

- SASRec 在此的角色是檢查共用資料與評估，不是追求最佳數字；不必為它調參。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

- 尚無阻礙；論文設定與 CE 的對照、多 seed 與最終 test 已列在 `TODO.md`。
