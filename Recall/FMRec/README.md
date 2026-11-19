# FMRec

FMRec（Flow Matching based Sequential Recommender）把下一件商品的預測當成 flow matching：從高斯雜訊出發，以使用者歷史為條件，沿 ODE 走到目標商品的 embedding，再以 embedding 相似度在全部商品中排名。論文：Feng Liu et al., **[Flow Matching based Sequential Recommender Model](https://arxiv.org/abs/2505.16298)**，IJCAI 2025；作者程式 [FengLiu-1/FMRec](https://github.com/FengLiu-1/FMRec)。

和 TIGER 不同，FMRec 不使用 SID，直接學 item ID embedding，讀取 Kuriyo 已準備好的 2014 Beauty ASIN 切分，所以不需要 `Representations/SID/`。

`Recall/FMRec/` 是一個模型家族。**每個變體自成一個完整空間**（程式、依賴、說明、測試、runs）；家族層只放這份說明與研究筆記 [INSIGHT.md](INSIGHT.md)。變體之間不互相 import，也不 import `Recall/TIGER/`；共用元件在 [`Recall/core/`](../core/README.md) 與 `core/`。

```text
Recall/FMRec/
├── README.md        # 本檔：家族說明與變體列表
├── INSIGHT.md       # 家族研究筆記（事實、限制、假設分開）
├── Experiment-log.md # 所有變體的實驗日誌（條目標明變體）
└── baseline/        # 依作者程式的重新實作（python -m Recall.FMRec.baseline.train）
    ├── README.md, requirements.txt, model.py, data.py, train.py, tests/
    └── runs/<RUN_NAME>/
```

## 變體

| 變體 | 狀態 | 說明 |
| --- | --- | --- |
| [`baseline/`](baseline/README.md) | 可執行；測試與 CPU smoke 通過，尚無完整訓練或 validation 結果 | 依作者程式 `1c39b17` 的 PyTorch 實作，使用 Kuriyo 20 件歷史上限；與作者程式的差異列在其 README |

新增變體的規則與 TIGER 家族相同：只有設定不同時用同一個變體加參數跑新的 run；變體有 run 記進 Experiment-log 後結果凍結，要改預設值、換演算法或修正會改變結果的 bug 時，複製成同層的新目錄，名稱描述差異，舊變體保留，並更新上表（規則見根目錄 `AGENTS.md`）。
