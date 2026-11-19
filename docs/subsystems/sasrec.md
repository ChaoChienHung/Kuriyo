---
title: SASRec baseline（資料與評估的檢查基準）
type: subsystem
created: 2026-11-19
updated: 2026-11-19
sources:
  - code: Recall/SASRec/README.md
  - code: Recall/SASRec/baseline/README.md
  - code: Recall/SASRec/Experiment-log.md
  - code: Recall/core/evaluation.py
  - code: Recall/SASRec/baseline/model.py
  - code: Recall/SASRec/baseline/data.py
  - code: Recall/SASRec/baseline/train.py
  - code: Recall/SASRec/baseline/tests/test_model.py
  - code: Recall/SASRec/baseline/tests/test_train.py
  - code: utils/amazon_reviews_2014.py
  - ext: https://arxiv.org/abs/1808.09781
verified_at: 4f6594d
---

# SASRec baseline（資料與評估的檢查基準）

`Recall/SASRec/baseline/` 是 SASRec（Kang & McAuley 2018）的 PyTorch 實作。它在 Kuriyo 的用途是當檢查基準：TIGER 換了健康的新 SID 之後，validation 仍和塌縮的舊 SID 差不多，所以要先確認共用的資料與評估沒有問題。TIGER 論文表 3 的 SASRec Beauty Recall@10 是 0.0605；Kuriyo 的 SASRec 在同一份切分與 `Recall/core/metrics.py` 下若接近這個值，差距就在 TIGER 自己的流程裡。結果記在 `Recall/SASRec/Experiment-log.md`，跨家族對照在 `Recall/INSIGHT.md`。

模型是 pre-norm Transformer encoder，加上因果遮罩、learned position embedding，並與 item embedding 共用輸出層（`Recall/SASRec/baseline/model.py:26`）。2026-11-19 起預設值是原論文設定：BCE、每位置一個負例且從使用者 train 序列沒出現過的商品中抽（`Recall/SASRec/baseline/data.py:69` 的 `sample_negatives`）、hidden 50、1 head；`--loss ce` 改成全 catalog softmax CE。評估取最後一個位置，對全 catalog 排名，不過濾歷史商品，和 TIGER、FMRec 一致（`Recall/SASRec/baseline/train.py:42`，排名本身在 `Recall/core/evaluation.py:11`）。與原論文的差異表在 baseline README。

訓練資料不是直接用前綴樣本，而是先還原每位使用者的序列，再切成視窗（`Recall/SASRec/baseline/data.py:9`、`Recall/SASRec/baseline/data.py:35`）。這樣每個位置都能算 loss，而每個訓練目標仍剛好用到一次，與 TIGER／FMRec 的訓練目標集合相同。

> ⚠️ `user_sequences` 依賴 train 檔的結構：同一位使用者的前綴樣本連續出現，從長度 1 的前綴開始，依時間排序（`utils/amazon_reviews_2014.py:43` 的 `build_examples`）。函式會逐筆檢查每筆的 history 是否等於已還原序列的尾端，不符合就報錯。改動切分格式或排序時，這裡會先失敗；不要為了讓它通過而移除檢查。

> ⚠️ 位置 embedding 依固定寬度 `max_history` 右對齊：最新商品永遠在最後一個位置。`forward` 拒絕寬度不等於 `max_history` 的輸入（`Recall/SASRec/baseline/model.py:47`）。若改成只 pad 到 batch 內最長長度，同一件商品在不同 batch 會拿到不同的位置 embedding，訓練和評估就對不起來。

> ⚠️ 左側 padding 加上因果遮罩時，padding 的 query 列若把所有 key 都遮掉，softmax 會得到 NaN。`attention_allowed` 讓 padding 位置只 attend 自己，真實位置則永遠不 attend padding（`Recall/SASRec/baseline/model.py:12`）。修改遮罩時要同時保住這兩點；測試 `test_real_positions_never_attend_padding` 與 `test_padding_positions_do_not_leak_into_real_outputs` 會檢查。

> ⚠️ 預設值在 2026-11-19 從 CE／hidden 64／2 heads 改成論文設定。之前的 run（`sasrec-beauty-v1` 與本機 run）續訓、test 或搬移都要帶 `--loss ce --hidden 64 --heads 2`，否則設定核對會拒絕。CE 路徑在改動前後以固定 seed 比對，loss 與 validation 逐位相同。

> ⚠️ `sample_negatives` 是拒絕抽樣：使用者看過全部商品時無法抽負例，函式會直接報錯而不是無窮迴圈。負例用訓練的 `sampler` generator，續訓時跟著 checkpoint 還原；改抽樣順序會讓續訓結果與不中斷訓練不同（`test_train_resume_and_held_out_test` 會檢查）。

相關：[[model-family-layout]]（家族與變體的目錄規則）、[[run-directory]]（run 目錄與共用訓練零件）、[[fmrec]]（另一個讀 ASIN 切分的家族）。
