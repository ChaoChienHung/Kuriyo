> 內容限制：本文件不得超過 **0.1M tokens（100,000 tokens）**。更新時以精簡、可驗證的狀態與來源連結為主，不貼完整 log、資料集或 checkpoint；過時狀態應修正，不無限追加聊天紀錄。

# Kuriyo 交接紀錄

更新日期：2026-11-19。先讀 [AGENTS.md](AGENTS.md)、[README.md](README.md)；未完成的工作見 [TODO.md](TODO.md)，等審查的工作見 [REVIEW.md](REVIEW.md)，研究結論與假設從 [INSIGHT.md](INSIGHT.md) 的各層連結開始。每個 run 的設定與完整數字記在變體的 `Experiment-log.md`（`Recall/<family>/baseline/`、`Representations/SID/`），本檔只留交接所需的摘要。

## 背景與目標

Kuriyo 是推薦模型研究 repo，預計包含 Recall、Ranking、其他推薦鏈路與端到端模型，也容納同一家族的實驗變體。近期目標是建立 Amazon Reviews 2014 Beauty 上的 TIGER baseline，逐項核對論文設定、盡可能重現數值；目前只是可執行的 PyTorch/Hugging Face 實作，不能稱為已完成論文重現。

每個模型變體是家族下自成一體的完整目錄（程式、依賴、README、tests、runs），變體之間不互相 import，只共用 `Recall/core/`、`core/` 等跨模型元件；只有設定不同時使用具名設定。目前只有 `Recall/TIGER/baseline/`（入口 `python -m Recall.TIGER.baseline.train`，原 `Recall/TIGER/` 下的 sequence、smoke trainer、config、requirements、tests 全部搬入，程式邏輯不變）；家族根層只剩 README。

## Repo 分工與保留原則

- `core/`：跨鏈路工具，目前是 `RunLogger`；`Recall/core/`：召回模型共用 padding、metrics、Transformer、實驗工具。
- `Representations/SID/`：獨立商品表示、Sentence-T5/RQ-VAE，不隸屬 TIGER。
- `Recall/TIGER/`：家族 README；`Recall/TIGER/baseline/`：SID tokenization、解碼、smoke trainer、paper-configured trainer、requirements、tests 與 runs。
- `scripts/`：資料下載／準備／轉換、Slurm orchestration、舊 log 匯入；兩個 `train_tiger_*` 是保留的相容入口。
- `tests/`：跨階段與資料元件；`Recall/tests/`：共用 Recall 行為；`Recall/TIGER/baseline/tests/`：baseline 的解碼、資料載入與訓練／續訓／test 整合。
- `Ranking/`、`pipelines/` 尚為規劃／說明，不是已完成模型。
- 原始資料、SID、checkpoints、環境與 cache 不入 Git。不刪除前次正式實驗來「整理」目錄；smoke 產物只用來確認流程，看完結果就可刪除，也不寫進 Experiment-log。

## 資料與切分

標準路徑：`data/amazon_reviews/<year>/raw/<category>/`、`catalog/<category>/`、`<train|validation|test>/raw/<category>/`。

2014 Beauty：22,363 users、12,101 items；131,413 train、22,363 validation、22,363 test。依 user 時間排序，來源行號打破同日 ties，最後一筆 test、倒數第二筆 validation，其餘 train prefix expansion；歷史最多 20 items。9,719 位 user 的 validation/test 同日，不能把來源順序當真實日內順序。

完整固定商品目錄生成 SID（包括 validation/test 商品 metadata）；不等同 cold-start 協定。商品 SID 放 `2014/artifacts/sids/sentence-t5_rqvae/Beauty/<version>/`，三個 split 的轉換資料放各自 `sids/<method>/Beauty/<version>/`。2023 All_Beauty 只有下載流程，尚未建立獨立評估協定。

## 已完成的完整 run 與證據

來源：`slurm-kuriyo-tiger-1153307.out`；已於 `2ad21c7` 從 repo 移除，可用 `git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out` 取回。job `1153307`；訓練程式 revision `5f99955`；SID_VERSION 與 RUN_NAME 都是 `tiger-paper-20261117-005232`。`STAGE=all` 完成 SID 20,000 epochs、split materialization、TIGER 200,000 steps，log 最後有 Saved 訊息；尚未取得獨立 sacct 成功狀態或實際 checkpoint 校驗。

| 項目 | Log 中觀察 |
|---|---|
| SID catalog | 12,101 商品 |
| codebook 使用碼數 | 6 / 26 / 25，各層容量 256（約 2.34% / 10.16% / 9.77%） |
| 碰撞群組 | 228；不是碰撞商品比例，不能直接當 collision rate |
| RQ-VAE loss | epoch 1 約 8.744e9；epoch 20,000 為 0.178358 |
| T5 參數量 | 5,587,968；取代舊文件「約 470 萬」估值 |
| 最佳已記錄 validation step | 200,000（按 NDCG@10） |
| Recall@5 / Recall@10 | 0.020569691 / 0.034208290 |
| NDCG@5 / NDCG@10 | 0.012409267 / 0.016787758 |
| invalid beam fraction | 十次 validation 都為 0 |

尚無此 run 的最終 test 結果。不能把 validation 與論文 test 直接比較，也不能宣稱低碼本使用率已被證明是差距原因。低使用率及第一 epoch 巨大 loss 是下一步的調查線索。

baseline run 現在的標準位置是 `Recall/TIGER/baseline/runs/<RUN_NAME>/`。SoC 上的 `tiger-paper-20261117-005232` 仍在舊路徑 `Recall/TIGER/runs/`，續訓或 test 前需 `mkdir -p Recall/TIGER/baseline/runs && mv Recall/TIGER/runs/tiger-paper-20261117-005232 Recall/TIGER/baseline/runs/`（尚未執行；`run.sbatch` 遇到只在舊路徑的 run 會拒絕執行）。SID 產物路徑不變；本地只有舊 smoke 產物、git 歷史中的 Slurm log 及重建的匯入紀錄，不假設已同步完整 GPU checkpoints／manifest。

## Run 目錄（2026-11-19 起）

SASRec、FMRec、TIGER baseline 的 run 改用 `core/runs.py` 的版面：`settings.json`（資料與前處理、評估集、模型、訓練、seed、revision）、`events.log`、`train/train.jsonl`、`checkpoints/{latest,best}.pt`、`validation/amazon2014-beauty/`、`test/amazon2014-beauty/metrics.json`（見 [core/README.md](core/README.md#run-目錄)）。重構前後以固定 seed 的小 run 比對，loss 與 validation／test 指標逐位相同。

SoC 上的既有 run（`tiger-t5x-adafactor-v1`、`tiger-paper-v1`、`sasrec-beauty-v1`、`fmrec-beauty-v1` 等）都是舊的平版面。新程式在 `--resume` 或 `--eval-test-only`（`STAGE=train` 續訓或 `STAGE=test`）時先用 `config.json` 核對設定，相符才搬進新版面並寫 `settings.json`。執行中的 job 1165519 已載入舊程式，pull 不影響它；它還在跑時，不要對同一 run 執行新程式。

## 多資料集、多 seed 與統計規則（2026-11-19）

`scripts/run.sbatch` 新增 `CATEGORY`（`Beauty` 預設、`Sports_and_Outdoors`、`Toys_and_Games`）、`SEED`（預設 42）與 `TRAIN_ARGS`（單一因素的訓練旗標，train 與 test 都帶上）。`scripts.ensure_amazon_reviews_2014 --category <分類|all>` 取代只做 Beauty 的準備腳本（舊名保留為 wrapper）。比較與下結論的 seed 數與檢定方式見 README「實驗設計與統計檢定」；逐使用者配對檢定需要的排名輸出還沒做（TODO）。SoC 上兩個新分類尚未準備；既有的 `sasrec-beauty-v1`、`fmrec-beauty-v1` 都是 seed 42、尚未跑 test。

**SASRec 預設改成論文設定（2026-11-19）：**BCE、負例排除看過的商品、hidden 50／1 head；`MODEL=sasrec` 的預設 RUN_NAME 改為 `sasrec-paper-<分類>-v1`。`sasrec-beauty-v1` 是舊預設（CE），續訓或 test 要帶 `TRAIN_ARGS="--loss ce --hidden 64 --heads 2"`。它的第一次 test（job 1166757，程式 `4f6594d`）在評估前以 "best checkpoint missing or test already evaluated" 結束；同樣的平版面在本機會正常搬移並完成 test，推測 job 的工作目錄不是放 run 的 repo（待使用者以 `sacct -j 1166757 --format=WorkDir` 確認）。後來確認是先前的提交已完成 test、1166757 是重送被拒。`sasrec-beauty-v1` test Recall@10 0.0851、`fmrec-beauty-v1` 0.0804（各 Experiment-log 2026-11-19 Held-out test）。`run.sbatch` 的 `STAGE=test` 現在先確認 best checkpoint 存在並印出完整路徑。

**Run 紀錄進 Git（2026-11-19）：**`.gitignore` 改成只追蹤 run 目錄的 `settings.json`、`events.log`、`validation/`、`test/`。在 SoC 上 `git add Recall/*/baseline/runs` 前先看 `git status`，只提交有 Experiment-log 條目的 run。

## 2026-11-19 晚：SASRec H1 成立、TIGER 20k 後下降

SASRec 論文設定（BCE）3 個 seed 的 validation Recall@10 0.0602 ± 0.0015，只換 CE 0.1024 ± 0.0006（jobs 1166902–07，程式 `1d5c37c`）；H1 成立。TIGER job 1165519 的 validation 在 20k 最好（0.0800），40k–100k 一路降到 0.0546，無效 beam 升到 1.8%；train loss 同時仍在降（30k 1.29），支持過擬合（TIGER INSIGHT H8）。使用者已結束該 job，`best.pt` 在 20k；其 test Recall@10 0.0556（job 1167548），比論文低約 14%、比 validation 低 31%。下一步：以 `--eval-every 2000 --steps 40000` 跑細曲線找最佳 step（TODO 第一項）。

## 分組評估（2026-11-19）

`STAGE=groups` 把既有 run 的 `best.pt` 依目標熱門度（train 當目標 0、1–4、5 以上次）重新評估，不訓練；有 test 的 run 一併拆解 test。下一步是對 SASRec、FMRec、TIGER 的既有 run 執行並記錄（TODO）。

## Logging 與匯入狀態

新版 `core/run_logging.py` 已接入 paper trainer 和 SID generator：TIGER 每 step 寫 run 的 `train/train.jsonl`、SID 每 epoch 寫 `train.jsonl`，生命週期等事件寫 `events.log`（JSON Lines），含 UTC timestamp、session ID、elapsed_seconds。續訓追加，不抹掉歷史；重跑相同 step 用 session ID 區分。SIGKILL、斷電或 logger 啟動前的錯誤未必留下失敗事件。舊 smoke trainer 尚未接入。

完整 run 使用的是新版 logger 之前的程式，因此不能回溯逐 step loss／時間。已用 `scripts/import_slurm_training_log.py` 在本地建立：

`Recall/TIGER/baseline/runs/tiger-paper-20261117-005232/imports/slurm-1153307/`

內含 tiger 的 2,001 training rows／14 events、sid 的 201 epoch rows／2 events，以及 import_manifest。來源 SHA、行號和 `origin=slurm_import` 可追溯；缺少的時間、SID learning rate 為 null。不可把匯入時間當訓練時間。此目錄被忽略，可重建：

```bash
mkdir -p .tmp && git show 2ad21c7^:slurm-kuriyo-tiger-1153307.out > .tmp/slurm-kuriyo-tiger-1153307.out
python3 -m scripts.import_slurm_training_log .tmp/slurm-kuriyo-tiger-1153307.out \
  --output-dir Recall/TIGER/baseline/runs/tiger-paper-20261117-005232/imports/slurm-1153307
```

已有目錄時拒絕覆寫；不要為重跑命令直接刪除已有紀錄。SoC 是否已執行匯入未確認。

## SoC 操作與 Git 注意事項

`run.sbatch` 支援 prepare / sid / materialize / train / test / all；all 不含最終 test。`MODEL=sasrec|fmrec` 改跑該家族 baseline（只接受 prepare / train / test / all，不用 SID）；同時提交多個 job 時用 `STAGE=train`，避免同時準備資料。SID 階段自動準備輸入，完整產物可重用，部分產物拒絕覆寫。TIGER 可從 latest.pt 續訓；RQ-VAE 尚無中途 checkpoint／resume。

目前 script 預設 gpu-long、1 GPU、32G、2 天；每次提交仍需依 [SoC skill](skills/nus-soc-slurm/SKILL.md) 確認 live 資源。除非使用者要求，不提交遠端作業或 push。

SoC 安裝指令在根 requirements.txt 註解與 run.sbatch，明確使用 torch 2.7.1+cu126。xgpd5 為 TITAN V/sm_70，曾因 CUDA 13 build 發生 no kernel image。login xlogin1 的 virtual-memory soft/hard limit 都曾是 1 GiB，import Torch mmap 失敗；在 compute allocation 內診斷，不把磁碟 quota 與記憶體限制混為一談。TMPDIR 使用 repo .tmp 處理安裝暫存空間。

曾有 SoC 未推送 commit 把舊 `data/amazon_reviews/{raw,train,eval,test}/<year>/` 納入，GitHub 因 >100 MB 拒收。已由使用者保留 backup branch、重整未推送 main；遠端 `679d990` 只含 ignore 與 log。新版 ignore 同時排除舊路徑；不 push backup branch 或使用 --all。目前沒有大型資料/checkpoint 進入遠端 main 的證據。不要要求分享 PAT；憑證只在使用者端設定。

## FMRec baseline（2026-11-18 新增）

`Recall/FMRec/baseline/` 依作者程式（FengLiu-1/FMRec `1c39b17`）重新實作，直接讀 2014 Beauty ASIN 切分、20 件歷史上限，以 `Recall/core/metrics.py` 做 full-ranking Recall／NDCG@5/10，可與 TIGER 同切分比較。時間取樣 s 預設 1.0（作者程式預設；論文 B.2 另提 0.4，可用 `--mode-scale`）。與作者程式的差異（padding key 遮罩、排除 padding 商品、以 NDCG@10 選模型等）列在其 README。完整 unittest discovery 43 項通過；本機 CPU smoke `fmrec-smoke-cpu-20261118`（4,096 train／1,024 validation、2 epochs）loss 有限，指標接近 0，不代表品質。尚未在 GPU 完整訓練；`run.sbatch` 以 `MODEL=fmrec` 提交，指令見其 README。

## SASRec 基準與 TIGER 新 SID（2026-11-19）

新 SID `tiger-paper-v1` 的 TIGER run（SoC job 1163052，log 在 `00d4470`）跑滿 200k，validation Recall@10 0.0368，舊 SID 為 0.0342，新 SID 幾乎沒有改善。為檢查共用資料與評估，新增 `Recall/SASRec/baseline/`；本機 MPS 完整 run `sasrec-beauty-v1`（約 23 分鐘，epoch 75 early stop）best epoch 50 的 validation Recall@10 0.0938、NDCG@10 0.0461，約 TIGER 的 3 倍。結論：切分、歷史上限與指標正常，TIGER 的差距在它自己的流程。完整數字在 `Recall/SASRec/INSIGHT.md` 與 `Recall/INSIGHT.md`。

程式碼檢查找到主因：TIGER 的 HF Adafactor 用 `scale_parameter=False`，lr 0.01 成了絕對步長；T5X（論文框架）預設乘上參數 RMS。新增 `--adafactor-scale-parameter`，SoC job 1165519 證實它讓 20k validation Recall@10 從 0.0132 變成 0.0800；**2026-11-19 起這是 `train.py` 與 `run.sbatch` 的預設**，續訓舊 run 需 `ADAFACTOR_SCALE_PARAMETER=0`（否則會被拒絕）。證據見 `Recall/TIGER/INSIGHT.md` H5。

## 下一步（依優先順序）

1. **驗證 Adafactor 假設（H5）：**在 SoC 執行
   `SID_RECIPE=stable SID_VERSION=tiger-paper-v1 RUN_NAME=tiger-t5x-adafactor-v1 ADAFACTOR_SCALE_PARAMETER=1 STAGE=train sbatch --export=ALL scripts/run.sbatch`。
   第一次提交 job 1165195 落在節點 xgpe2，CUDA 初始化失敗（`CUDA unknown error`），train.py 在建立 run 目錄前以 exit 2 結束，沒有訓練；重送時加 `--exclude=xgpe2`，同一 RUN_NAME 可用。
   2026-11-19 同時提交三個 job（程式 `bb1088c`）：TIGER job 1165519（`tiger-t5x-adafactor-v1`）與 FMRec job 1165520（`fmrec-beauty-v1`）正在訓練；SASRec job 1165521（xgpe4）與重送的 1165548（xgpe6）都因 `CUDA unknown error` 在建立 run 目錄前結束。sacct 確認 TIGER／FMRec 在 xgpf11 正常執行；xgpe2／4／6 在 gpu-long 上顯示 idle，所以新 job 容易先分到它們。SASRec 重送時用 `--exclude='xgpe[0-4,6]'` 排除整組 xgpe（xgpe5 不存在，寫 `xgpe[0-6]` 會被 sbatch 以 Invalid node name 拒絕）。
   之後排除 xgpe 重送的 SASRec job 1165570 已跑完（best validation Recall@10 0.1023，本機為 0.0938）；FMRec 在 epoch 260 early stop，best epoch 160 的 Recall@10 0.1031。**TIGER job 1165519 在 20k 的 validation Recall@10 0.0800（baseline 同 step 0.0132），H5 強烈支持**；job 繼續跑滿 200k。數字見各變體的 Experiment-log。
   比較 20k／40k validation 與 `tiger-paper-v1`（Recall@10 0.0132／0.0174）。明顯較高就跑滿 200k；否則依 `Recall/TIGER/INSIGHT.md` Problems 2 繼續查。
2. 新 SID 完整 run `tiger-paper-v1` 已完成（200k Recall@10 0.0368），SID 產生紀錄見 job 1162430 與 `Representations/SID/INSIGHT.md`。已記錄進 INSIGHT 的 Slurm log（job 1162430、1163050、1163052）與失敗 job（1165195、1165521）的 log 已從工作目錄移除，可用 `git show b7a4078:slurm-kuriyo-<job>.out` 取回。
3. 核對 paper/T5X 與 HF T5 的容量、商品文字、user hash、訓練樣本、optimizer、解碼與評估設定；將事實與猜測分開記錄。
4. 保留本 run 作比較起點，新增實驗用新 SID_VERSION/RUN_NAME；固定 validation，先建立熱門商品基線。模型定案後才執行一次 held-out test。
5. **FMRec 完整訓練：**在 SoC GPU 執行 `MODEL=fmrec RUN_NAME=fmrec-beauty-v1 STAGE=train sbatch --export=ALL --job-name=kuriyo-fmrec scripts/run.sbatch`（作者預設，約 257 steps/epoch、最多 500 epochs；約 2,840 萬參數，CPU 不可行）。validation 與 SASRec、TIGER、熱門基線比較；之後的單一因素對照（`--mode-scale 0.4`、`--attention causal`、取樣步數）見 TODO「FMRec 同切分比較」。
6. 增加 SID checkpoint/resume；有第二個實際模型變體時，於 `Recall/TIGER/<variant-name>/` 新增並視需要建立載入介面。

## 驗證與限制

最近已執行：22 項核心／TIGER／logging 測試通過，SID 8-item TF-IDF 小跑驗證新 logger；之後新增的 2 項 Slurm importer 測試通過，真實 log 筆數、最後 step、10 次 validation 已核對。TIGER 改為 `baseline/` 自成空間後，完整 unittest discovery 26 項全數通過（含 run.sbatch 舊路徑保護測試）；文件本地連結與 git diff --check 通過。

```bash
.venv/bin/python -m unittest discover -s . -v
```

測試使用 CPU／合成小資料，不等同 GPU 效能或論文重現。本次不改模型／切分／訓練參數、不重跑完整 GPU 訓練。
