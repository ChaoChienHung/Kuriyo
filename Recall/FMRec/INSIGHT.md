# FMRec — Research Notes

FMRec 家族的研究筆記。事實與假設分開，每個結果都註明資料版本、切分、模型設定、指標，以及來自哪一條 Experiment-log 或哪個 run。操作方式見 [README](README.md) 與 [baseline README](baseline/README.md)；跨家族比較在 [Recall INSIGHT](../INSIGHT.md)。

*建立：2026-11-18；最後更新：2026-11-19。Experiment log：[baseline](baseline/Experiment-log.md)。*

## Insights

從資料或實驗得到、會改變後續做法的解讀。

- 尚無：還沒有完整訓練的 validation 結果。

## Facts

已確認的資料性質、實作細節與 run 結果。

- 作者程式 [FengLiu-1/FMRec](https://github.com/FengLiu-1/FMRec)（核對 commit `1c39b17`）的預設值：hidden 128、4 個 Transformer block（`last=2`，即論文的 Decoder1／Decoder2 各 2 層）、4 heads、dropout 0.1、embedding dropout 0.3、`lambda_uncertainty` 0.001、eps 0.001、Euler 30 步、mode sampling `s_modsamp=1.0`、α=0.2、β=0.4、Adam lr 1e-3、StepLR 每 100 epochs ×0.1、batch 512、500 epochs、每 20 epochs 評估、patience 5、max_len 50、seed 1997。
- 作者程式做 x₀ 預測，z_t = t·x_target + (1−t)·noise（t=1 是資料）；L_FM 是 x₀ 預測對目標 embedding 的 MSE，L_MSE 是從 Decoder1 最後位置經 MLP 128→512→2048→N 重建歷史 multi-hot 的 balanced MSE。
- 作者程式沒有位置 embedding、沒有因果遮罩；attention mask 遮的是 padding 的 query 列，真實位置仍 attend padding key。Kuriyo baseline 改為遮 key（見 baseline README 的差異表）。
- 論文 Beauty 結果（HR@5 5.84%、HR@10 8.27%、NDCG@5 4.16%、NDCG@10 4.95%）使用 50 件歷史、ICLRec／DuoRec 處理的資料（22,361 users、12,101 items），論文沒有和 TIGER 比較。
- Kuriyo 的 20 件歷史上限來自 TIGER 論文 Appendix C（套用在訓練）；FMRec 論文用 50。Kuriyo 2014 Beauty 切分：22,363 users、12,101 items。
- Kuriyo baseline 28,388,038 參數（SoC job 1165520 log），其中約 2,480 萬在重建 MLP 最後一層（2048 × 12,102）。
- CPU smoke `fmrec-smoke-cpu-20261118` 只證明流程可執行（[baseline Experiment-log](baseline/Experiment-log.md) 2026-11-18）；完整訓練 `fmrec-beauty-v1` 在 SoC 進行中（同檔 2026-11-19）。

## Limitations

會限制結論範圍的已知弱點。

- 尚無完整訓練、validation 或 test 結果。
- 20 件歷史上限與論文的 50 不同；資料前處理也不同，因此即使跑完，也不能直接和論文數字比較。
- 論文 Appendix B.2 對 s 有 1.0 與 0.4 兩種說法；目前只依作者程式預設 1.0。
- 評估不過濾歷史商品（與作者程式、TIGER baseline 一致）。

## Hypothesis

待驗證的主張。編號固定；跨檔引用寫 `Recall/FMRec/INSIGHT.md` H[n]。

- **H1 — 取樣步數幾乎不影響結果：**x_t 只以約 0.001 的權重注入，預測主要由歷史決定；Euler 30 步的結果可能與單步預測幾乎相同。驗證：比較 `--sample-steps 1` 與 30 的 validation。
- **H2 — FMRec 優於 TIGER baseline：**在相同切分下，FMRec（全排名、item ID embedding）的 validation Recall@10 會高於 TIGER baseline 的 0.0342／0.0368。驗證中：[baseline Experiment-log](baseline/Experiment-log.md) 2026-11-19（job 1165520）。

## Discussion

仍在權衡的設計選擇。

- FMRec 不需 SID，可作為 TIGER 的非生成式 SID 對照：若 FMRec 明顯較好，TIGER 的差距較可能來自 SID 品質或生成解碼，而非切分或指標。

## Problems to solve

具體的阻礙或未知；排入工作後在 `TODO.md` 追蹤。

- 等完整訓練的 validation（job 1165520），與 TIGER、SASRec、熱門基線比較。
