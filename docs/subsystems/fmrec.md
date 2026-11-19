---
title: FMRec baseline（flow matching 序列推薦）
type: subsystem
created: 2026-11-18
updated: 2026-11-18
sources:
  - code: Recall/FMRec/README.md
  - code: Recall/FMRec/baseline/README.md
  - code: Recall/FMRec/baseline/model.py
  - code: Recall/FMRec/baseline/data.py
  - code: Recall/FMRec/baseline/train.py
  - code: Recall/FMRec/baseline/tests/test_model.py
  - code: Recall/FMRec/baseline/tests/test_train.py
  - ext: https://arxiv.org/abs/2505.16298
  - ext: https://github.com/FengLiu-1/FMRec
verified_at: 2c946fe
---

# FMRec baseline

`Recall/FMRec/baseline/` 是 FMRec（Liu et al., IJCAI 2025）的 PyTorch 重新實作，目的是在 Kuriyo 的 2014 Beauty leave-one-out 切分上，用和 TIGER 相同的指標函式比較兩者。它不使用 SID，直接學 item ID embedding，讀 `data/amazon_reviews/2014/<split>/raw/Beauty/examples.jsonl`。目錄規則見 [[model-family-layout]]；操作指令、預設值與差異表在 `Recall/FMRec/baseline/README.md`，本頁只記結構與坑。

## 流程

- `Recall/FMRec/baseline/data.py`：catalog 順序決定商品 index（1..N，0 是 padding）；`collate` 把最近 `max_history` 件左側補零到批次最大長度（`Recall/FMRec/baseline/data.py:38`）。
- `Recall/FMRec/baseline/model.py`：`FMRec.loss` 抽時間 t、組 z_t、做 x₀ 預測，合併 L_FM + α·L_CE + β·L_MSE（`Recall/FMRec/baseline/model.py:189`）；`FMRec.sample` 以 Euler 30 步從雜訊走到 z₁（`Recall/FMRec/baseline/model.py:204`）；`FMRec.scores` 以內積對全 catalog 排名（`Recall/FMRec/baseline/model.py:167`）。
- `Recall/FMRec/baseline/train.py`：epoch 制訓練、每 `--eval-every` epochs 做 validation、以 NDCG@10 選 `best.pt`、每 epoch 寫 `latest.pt`，`--eval-test-only` 只跑一次 test（`Recall/FMRec/baseline/train.py:107`）。

## 對照作者程式時的事實

以下由作者程式 commit `1c39b17` 確認（論文文字與程式不一致時以程式為準）：

- 時間方向：z_t = t·x_target + (1−t)·noise，t=1 是資料；網路預測 x₀，不是速度。
- 論文的 Decoder1／Decoder2 是同一個 4 層 stack 的前後兩段（`num_blocks=4, last=2`）；Decoder1 的最後位置接歷史 multi-hot 重建 MLP，這就是 L_MSE。
- x_t 只以約 0.001 的權重加到歷史表示上（`lambda_uncertainty`），所以預測主要由歷史決定。
- 沒有位置 embedding；作者程式也沒有因果遮罩。
- mode sampling 的 s 預設 1.0（論文 Appendix B.2 另提 0.4）。

> ⚠️ 作者程式的 attention mask 把 mask 廣播到 **query** 維度，等於遮掉 padding 的輸出列，真實位置仍會 attend padding key。照抄會讓預測隨 padding 長度改變。本實作改成遮 key，並允許每個位置 attend 自己，避免全 padding 列在 causal 模式下變成 NaN（`Recall/FMRec/baseline/model.py:29`）。改 attention 時要保留 `test_left_padding_does_not_change_prediction`。

> ⚠️ 沒有位置 embedding，又是雙向 attention，模型只能從「最後一個位置」這件事分辨順序。這是作者的設定，不是漏寫；加位置 embedding 或改成 `--attention causal` 都是另一個實驗，要開新 run（模型設定寫在 run 目錄 config.json 的 `model` 欄，續訓時不符會拒絕）。

> ⚠️ uncertainty 權重是對 `[B, L, H]` 逐元素抽樣，抽樣結果會隨 padding 寬度不同而對到不同位置，所以「padding invariance」只在分布上成立。測試把這項雜訊固定在平均值，才能逐值比較（`Recall/FMRec/baseline/tests/test_model.py`）。

> ⚠️ 評估雜訊（z₀ 與 uncertainty）用 `--seed` 建立的獨立 CPU generator 產生，不動全域 RNG（`Recall/FMRec/baseline/model.py:133`、`Recall/FMRec/baseline/train.py:35`）。若改回 `torch.randn` 直接抽，validation 會消耗訓練 RNG，續訓與不中斷訓練的 loss 會分岔，`test_train_resume_and_held_out_test` 會失敗。

> ⚠️ `FMRec.scores` 把 padding 欄設為 −inf，CE 與排名都排除 index 0。作者程式沒有排除。`evaluate` 用 `asins[index - 1]` 對回 ASIN，若哪天 padding 欄沒被排除，index 0 會被對到最後一件商品，錯誤不會報錯。

> ⚠️ 參數量約 2,840 萬，大部分在重建 MLP 最後一層（2048 × 商品數）。CPU 上每 step 很慢，完整訓練（約 257 steps/epoch，最多 500 epochs）要用 GPU。

## 與 TIGER 的可比性

兩者讀同一份切分（20 件歷史上限）並用同一個 `Recall/core/metrics.py`。差別：TIGER 用 beam search 產生 top-10，無效 SID 算 miss；FMRec 對 12,101 件全排名。兩者都不過濾歷史商品。FMRec 論文的數字使用 50 件歷史與它自己的切分，不能直接和這裡的 validation 比較。
