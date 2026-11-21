# FMRec — Research Notes

FMRec 家族的研究筆記。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標，以及來自哪一條 Experiment-log 或哪個 run。操作方式見 [README](README.md) 與 [baseline README](baseline/README.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

*建立：2026-11-18；最後更新：2026-11-21。Experiment log：[Experiment-log.md](Experiment-log.md)（家族所有變體）。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- 在同一切分與指標下，FMRec 的 best validation Recall@10 0.1031 與 SASRec 相當，NDCG@10 0.0603 比 SASRec 高約 0.010：命中的商品排得更前面。又一個 item-ID 模型正常，TIGER 的差距確定在 TIGER 自己的流程。證據：[baseline Experiment-log](Experiment-log.md) 2026-11-19。

## Facts

已確認的資料性質、實作細節與 run 結果。

- 作者程式 [FengLiu-1/FMRec](https://github.com/FengLiu-1/FMRec)（核對 commit `1c39b17`）的預設值：hidden 128、4 個 Transformer block（`last=2`，即論文的 Decoder1／Decoder2 各 2 層）、4 heads、dropout 0.1、embedding dropout 0.3、`lambda_uncertainty` 0.001、eps 0.001、Euler 30 步、mode sampling `s_modsamp=1.0`、α=0.2、β=0.4、Adam lr 1e-3、StepLR 每 100 epochs ×0.1、batch 512、500 epochs、每 20 epochs 評估、patience 5、max_len 50、seed 1997。
- 作者程式做 x₀ 預測，z_t = t·x_target + (1−t)·noise（t=1 是資料）；L_FM 是 x₀ 預測對目標 embedding 的 MSE，L_MSE 是從 Decoder1 最後位置經 MLP 128→512→2048→N 重建歷史 multi-hot 的 balanced MSE。
- 作者程式沒有位置 embedding、沒有因果遮罩；attention mask 遮的是 padding 的 query 列，真實位置仍 attend padding key。Kuriyo baseline 改為遮 key（見 baseline README 的差異表）。
- 論文 Beauty 結果（HR@5 5.84%、HR@10 8.27%、NDCG@5 4.16%、NDCG@10 4.95%）使用 50 件歷史、ICLRec／DuoRec 處理的資料（22,361 users、12,101 items），論文沒有和 TIGER 比較。
- `fmrec-beauty-v1`（作者預設，seed 42）在 Kuriyo 切分的 test：Recall@5 0.0576、Recall@10 0.0804、NDCG@5 0.0402、NDCG@10 0.0474，四項都在論文數字的 0.0025 以內；資料處理與歷史長度不同，只能當參考。證據：[Experiment-log](Experiment-log.md) 2026-11-19 Held-out test。
- 依目標熱門度分組，FMRec 在冷門目標（train 當目標 1–4 次）上是三個模型中最強：validation NDCG@10 0.0213（SASRec CE 0.0141、TIGER 0.0060）。對 SASRec CE 的逐使用者配對（單一 seed）：NDCG@10 在 validation 與 test 都較高（+0.0095／+0.0061，p < 1e-12），Recall@10 相當（validation +0.0008，p = 0.62；test −0.0047）。優勢在排序位置而非命中與否。證據：[Experiment-log](Experiment-log.md) 2026-11-20。
- 作者預設 5 個 seed（42–46）的 validation：Recall@10 0.1034 ± 0.0016（95% CI 0.1013–0.1054）、NDCG@10 0.0602 ± 0.0017。與 SASRec 只換 CE 在共同 seed 42–44 比較：NDCG@10 +0.0103，配對 t p = 7e-47、Welch p = 6e-5，依 README 判定「顯著較好」；Recall@10 +0.0015，兩種檢定都不顯著（p = 0.2／0.11），判定沒有差異。證據：[Experiment-log](Experiment-log.md) 2026-11-21。
- Kuriyo 的 20 件歷史上限來自 TIGER 論文 Appendix C（套用在訓練）；FMRec 論文用 50。Kuriyo 2014 Beauty 切分：22,363 users、12,101 items。
- Kuriyo baseline 28,388,038 參數（SoC job 1165520 log），其中約 2,480 萬在重建 MLP 最後一層（2048 × 12,102）。
- 完整訓練 `fmrec-beauty-v1`（SoC job 1165520，程式 `bb1088c`）在 epoch 260 early stop；best epoch 160 的 validation Recall@5 0.0728、Recall@10 0.1031、NDCG@5 0.0506、NDCG@10 0.0603。epoch 80 之後幾乎持平（[baseline Experiment-log](Experiment-log.md) 2026-11-19）。

## Limitations

會限制結論範圍的已知弱點。

- 只有一個 seed、只有 validation，沒有 test 結果。
- 20 件歷史上限與論文的 50 不同；資料前處理也不同，因此即使跑完，也不能直接和論文數字比較。
- 論文 Appendix B.2 對 s 有 1.0 與 0.4 兩種說法；目前只依作者程式預設 1.0。
- 評估不過濾歷史商品（與作者程式、TIGER baseline 一致）。
- 和 SASRec 一樣只用隨機初始化的商品 ID embedding（`L_FM` 的目標與最後的打分也是這些 embedding），對沒在 train 出現過的商品無能為力；目前的評估幾乎不測冷啟動（`Recall/INSIGHT.md` Limitations、H2）。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/FMRec/INSIGHT.md` H[n]。

- **H1 — 取樣步數幾乎不影響結果：**x_t 只以約 0.001 的權重注入，預測主要由歷史決定；Euler 30 步的結果可能與單步預測幾乎相同。驗證：比較 `--sample-steps 1` 與 30 的 validation。
- **H2 — FMRec 優於 TIGER baseline：**在相同切分下，FMRec（全排名、item ID embedding）的 validation Recall@10 會高於 TIGER baseline 的 0.0342／0.0368。支持（best 0.1031）：[baseline Experiment-log](Experiment-log.md) 2026-11-19（job 1165520）。

## Discussion

仍在權衡的設計選擇。

- FMRec 不需 SID，可作為 TIGER 的非生成式 SID 對照：若 FMRec 明顯較好，TIGER 的差距較可能來自 SID 品質或生成解碼，而非切分或指標。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

- 尚無阻礙；單一因素對照（H1 等）列在 `TODO.md`。
