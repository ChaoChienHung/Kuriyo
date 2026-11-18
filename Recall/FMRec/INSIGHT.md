# FMRec — Research Notes

FMRec 家族的研究筆記。`Facts` 是已由程式、資料或 run 確認的內容，`Hypothesis` 是待驗證的主張，兩者分開記錄。操作方式見 [README](README.md) 與 [baseline README](baseline/README.md)。

*建立日期：2026-11-18。*

## Facts

- 作者程式 [FengLiu-1/FMRec](https://github.com/FengLiu-1/FMRec)（核對 commit `1c39b17`）的預設值：hidden 128、4 個 Transformer block（`last=2`，即論文的 Decoder1／Decoder2 各 2 層）、4 heads、dropout 0.1、embedding dropout 0.3、`lambda_uncertainty` 0.001、eps 0.001、Euler 30 步、mode sampling `s_modsamp=1.0`、α=0.2、β=0.4、Adam lr 1e-3、StepLR 每 100 epochs ×0.1、batch 512、500 epochs、每 20 epochs 評估、patience 5、max_len 50、seed 1997。
- 作者程式做 x₀ 預測，z_t = t·x_target + (1−t)·noise（t=1 是資料）；L_FM 是 x₀ 預測對目標 embedding 的 MSE，L_MSE 是從 Decoder1 最後位置經 MLP 128→512→2048→N 重建歷史 multi-hot 的 balanced MSE。
- 作者程式沒有位置 embedding、沒有因果遮罩；attention mask 遮的是 padding 的 query 列，真實位置仍 attend padding key。Kuriyo baseline 改為遮 key（見 baseline README 的差異表）。
- 論文 Beauty 結果（HR@5 5.84%、HR@10 8.27%、NDCG@5 4.16%、NDCG@10 4.95%）使用 50 件歷史、ICLRec／DuoRec 處理的資料（22,361 users、12,101 items），論文沒有和 TIGER 比較。
- Kuriyo 的 20 件歷史上限來自 TIGER 論文 Appendix C（套用在訓練）；FMRec 論文用 50。Kuriyo 2014 Beauty 切分：22,363 users、12,101 items。
- Kuriyo baseline 約 2,840 萬參數，其中約 2,480 萬在重建 MLP 最後一層（2048 × 12,102）。
- 2026-11-18 CPU smoke `fmrec-smoke-cpu-20261118`（4,096 train／1,024 validation、2 epochs）loss 有限；指標接近 0，只證明流程可執行。

## Limitations

- 尚無完整 GPU 訓練、validation 或 test 結果。
- 20 件歷史上限與論文的 50 不同；資料前處理也不同，因此即使跑完，也不能直接和論文數字比較。
- 論文 Appendix B.2 對 s 有 1.0 與 0.4 兩種說法；目前只依作者程式預設 1.0。
- 評估不過濾歷史商品（與作者程式、TIGER baseline 一致）。

## Hypothesis

- H1：x_t 只以約 0.001 的權重注入，預測主要由歷史決定；Euler 30 步的結果可能與單步預測幾乎相同。待驗證：比較 `--sample-steps 1` 與 30 的 validation。
- H2：在相同切分下，FMRec（全排名、item ID embedding）的 validation Recall@10 會高於 TIGER baseline 的 0.0342。尚無證據。

## Discussion

- FMRec 不需 SID，可作為 TIGER 的非生成式 SID 對照：若 FMRec 明顯較好，TIGER 的差距較可能來自 SID 品質或生成解碼，而非切分或指標。

## Problems to solve

- 在 SoC GPU 以作者預設完整訓練，記錄 validation，與 TIGER、熱門基線比較。
- 單一因素對照：`--mode-scale 0.4`、`--attention causal`、50 件歷史（需另建切分）。
- 檢驗 H1（取樣步數敏感度）。
