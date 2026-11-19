# SASRec

SASRec（Self-Attentive Sequential Recommendation）用因果 self-attention 讀取使用者的商品 ID 序列，在每個位置預測下一件商品，再以商品 embedding 內積對整個 catalog 排名。論文：Wang-Cheng Kang、Julian McAuley，**[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)**，ICDM 2018。

在 Kuriyo 裡，SASRec 的角色是**檢查資料與評估的基準**：[TIGER 論文](https://arxiv.org/abs/2305.05065) 表 3 列出 SASRec 在 Beauty 上的 Recall@10 是 0.0605。若 Kuriyo 的 SASRec 在同一份切分與同一個指標函式下也接近這個數字，代表資料與評估沒問題，TIGER 的差距在它自己的流程裡；若也只有 0.03 左右，問題就在共用的資料或評估。它不使用 SID，直接讀 Kuriyo 的 2014 Beauty ASIN 切分。

`Recall/SASRec/` 是一個模型家族。**每個變體自成一個完整空間**（程式、依賴、說明、測試、runs）；家族層只放這份說明與研究筆記 [INSIGHT.md](INSIGHT.md)。變體之間不互相 import，也不 import 其他家族；共用元件在 [`Recall/core/`](../core/README.md) 與 `core/`。

```text
Recall/SASRec/
├── README.md        # 本檔：家族說明與變體列表
├── INSIGHT.md       # 家族研究筆記
└── baseline/        # PyTorch 實作（python -m Recall.SASRec.baseline.train）
    ├── README.md, Experiment-log.md, requirements.txt, model.py, data.py, train.py, tests/
    └── runs/<RUN_NAME>/
```

## 變體

| 變體 | 狀態 | 說明 |
| --- | --- | --- |
| [`baseline/`](baseline/README.md) | 可執行；本機完整 run validation Recall@10 0.0938 | 原論文架構，預設 full-softmax CE loss（`--loss bce` 可改回原論文的單一負例），使用 Kuriyo 20 件歷史上限 |

新增變體的規則與 TIGER 家族相同：只有超參數不同時用具名設定或命令列參數；確實需要不同實作才複製 `baseline/` 成同層目錄，名稱描述差異，並更新上表。
