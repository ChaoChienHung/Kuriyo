# Recall core

此目錄放不同召回模型共用的程式：[`metrics.py`](metrics.py) 提供下一商品任務的 Recall@K 與 NDCG@K，[`data.py`](data.py) 提供 token padding，[`experiment.py`](experiment.py) 提供 SHA、seed 與 run 紀錄，[`transformer.py`](transformer.py) 提供基本 encoder–decoder Transformer。TIGER 的小模型使用這些元件，測試位於 [`Recall/tests/`](../tests/)。候選商品輸出格式與共用訓練介面尚未定義。
