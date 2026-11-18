---
title: SID 的 RQ-VAE 訓練與已知失效模式
type: subsystem
created: 2026-11-18
updated: 2026-11-18
sources:
  - code: Representations/SID/rqvae.py
  - code: Representations/SID/scripts/generate_sids.py
  - code: Representations/SID/scripts/diagnose_rqvae.py
  - code: INSIGHT.md
  - code: scripts/run.sbatch
  - code: tests/test_sid_components.py
verified_at: 6331c6d
---

# SID 的 RQ-VAE 訓練與已知失效模式

`Representations/SID/scripts/generate_sids.py` 先用 Sentence-T5 把商品文字編碼成 768 維向量，並做 L2 normalize（`Representations/SID/scripts/generate_sids.py:61`）。接著訓練 `Representations/SID/rqvae.py` 的三層 RQ-VAE（每層 256 碼、latent 32 維），把每件商品量化成三個碼，再依 ASIN 排序附加第四個碰撞 token。

完整訓練以具名配方 `--recipe` 執行（`Representations/SID/scripts/generate_sids.py:38`）。三個配方共用論文的架構、20,000 epochs、batch 1024，以及從隨機 1,024 件商品做 k-means 初始化碼本，只差在穩定化方式：
- `paper-pytorch`：舊行為，重現 job 1153307；`--paper-baseline` 是它的別名。
- `paper-tf-adagrad`：Adagrad accumulator 0.1，加上重設未使用碼。
- `stable`：AdamW，加上 embedding 標準化與重設未使用碼。`scripts/run.sbatch` 預設使用它。

相關程式：
- Optimizer 由 `build_optimizer` 建立（`Representations/SID/scripts/generate_sids.py:120`）。
- 標準化由 `standardize` 處理（`Representations/SID/scripts/generate_sids.py:113`），平均值與標準差會存進 `embeddings.npz`。
- 重設未使用碼在 `Representations/SID/rqvae.py:79`。

`scripts/run.sbatch` 以 `SID_RECIPE` 選擇配方。重用既有 SID 目錄前會比對 manifest 的配方（`scripts/run.sbatch:69`）；舊 manifest 沒有 recipe 欄位，視為 `paper-pytorch`。

`Representations/SID/scripts/diagnose_rqvae.py` 讀取任一 SID run 的 `embeddings.npz`，用 `--recipe` 或個別選項重跑短訓練，逐段印出分項 loss、每層碼使用數與碰撞統計，不寫任何檔案。2026-11-18 的對照結果記在 `INSIGHT.md` 的 Facts／TIGER。

> ⚠️ PyTorch 的 `torch.optim.Adagrad` 預設 `initial_accumulator_value=0`，第一步的更新量約等於 `lr × sign(grad)`。lr 0.4 時，每個權重第一步就移動 0.4，latent 範數從 0.32 暴增到數千，三層碼本塌縮成同一個碼。論文的 lr 0.4 很可能來自 accumulator 預設 0.1 的框架（TF／Optax），照抄到 PyTorch 時必須明確設定 accumulator。SoC job 1153307 第一個 epoch 的 loss 8.7e9 和 6/26/25 的碼使用數，就是這個問題造成的。`build_optimizer` 一律明確傳入 accumulator；`tests/test_sid_components.py` 檢查 accumulator 0 會讓第一步 latent 範數暴增超過 10 倍，0.1 則不會。

> ⚠️ Sentence-T5 的 L2-normalized 向量有很大的共同分量：每維均方 0.0013，扣掉平均向量後只剩約 2.2e-4。RQ-VAE 只要輸出平均 embedding，就能把重建 MSE 降到約 2.2e-4，完全不需要用到碼本，於是第一層碼本退化成單一碼。看到重建 MSE 停在 2.2e-4 左右時，不能當作訓練成功；要同時看各層碼使用數與碰撞數。

> ⚠️ 碼本 loss 只會更新被選中的碼。latent 分布一旦移動，其他碼就再也選不到（dead code）。`reset_dead_codes` 每個 epoch 把未使用的碼重設為該層的 latent 殘差，才能維持 256/256/256 的使用率。重設時要從不同的商品取殘差（目前用 `randperm`）；若有放回抽樣，兩個碼可能被設成同一點，其中一個會立刻再次變成 dead code。修改量化流程時，要保留或替換這個機制。

> ⚠️ SID 訓練目前沒有中途 checkpoint，20,000 epochs 中斷就得重跑。改動訓練設定前，先用 `diagnose_rqvae` 跑 60 epochs 左右確認不塌縮，再提交長時間的 SoC job。

相關：[[model-family-layout]]（SID 由獨立元件產生，不屬於任何 TIGER 變體）。
