# SASRec

SASRec（Self-Attentive Sequential Recommendation）用因果 self-attention 讀取使用者的商品 ID 序列，在每個位置預測下一件商品，再以商品 embedding 內積對整個 catalog 排名。論文：Wang-Cheng Kang、Julian McAuley，**[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)**，ICDM 2018。

在 Kuriyo 裡，SASRec 的角色是**檢查資料與評估的基準**：[TIGER 論文](https://arxiv.org/abs/2305.05065) 表 3 列出 SASRec 在 Beauty 上的 Recall@10 是 0.0605。若 Kuriyo 的 SASRec 在同一份切分與同一個指標函式下也接近這個數字，代表資料與評估沒問題，TIGER 的差距在它自己的流程裡；若也只有 0.03 左右，問題就在共用的資料或評估。它不使用 SID，直接讀 Kuriyo 的 2014 Beauty ASIN 切分。

`Recall/SASRec/` 是一個模型家族。**每個變體自成一個完整空間**（程式、依賴、說明、測試、runs）；家族層只放這份說明與研究筆記 [INSIGHT.md](INSIGHT.md)。變體之間不互相 import，也不 import 其他家族；共用元件在 [`Recall/core/`](../core/README.md) 與 `core/`。

```text
Recall/SASRec/
├── README.md        # 本檔：家族說明與變體列表
├── INSIGHT.md       # 家族研究筆記
├── Experiment-log.md # 所有變體的實驗日誌（條目標明變體）
└── baseline/        # PyTorch 實作（python -m Recall.SASRec.baseline.train）
    ├── README.md, requirements.txt, model.py, data.py, train.py, tests/
    └── runs/<RUN_NAME>/
```

## 變體

| 變體 | 狀態 | 說明 |
| --- | --- | --- |
| [`baseline/`](baseline/README.md) | 可執行；論文設定尚未完整訓練；CE 設定 SoC validation Recall@10 0.1023 | 預設為原論文設定（BCE、負例排除看過的商品、hidden 50／1 head）；`--loss ce --hidden 64 --heads 2` 是先前的 CE 設定；使用 Kuriyo 20 件歷史上限 |

新增變體的規則與 TIGER 家族相同：只有設定不同時用同一個變體加參數跑新的 run；變體有 run 記進 Experiment-log 後結果凍結，要改預設值、換演算法或修正會改變結果的 bug 時，複製成同層的新目錄，名稱描述差異，舊變體保留，並更新上表（規則見根目錄 `AGENTS.md`）。
