# AI CUP 2025 玉山人工智慧公開挑戰賽 — Preprocess/data_preprocess.py

## 檔案說明
`Preprocess/data_preprocess.py` 是本專案的資料預處理與帳戶特徵工程模組。它會讀取比賽提供的三個 CSV、統一不同的欄位名稱，並把「交易層級」資料整合成「帳戶層級」的特徵，提供 `Model/graph_based_model.py` 的 GraphSAGE 模型直接使用。亦提供隨機種子設定以確保實驗可重現。

---

## 主要功能
- 載入資料：`acct_transaction.csv`、`acct_alert.csv`、`acct_predict.csv`
- 辨識欄位資訊（大小寫/別名/子字串比對），統一為：  
  `src`, `dst`, `amt`, `from_type`, `to_type`, `alert_acct`, `predict_acct`
- 交易資料處理 → 帳戶特徵整合：轉入/轉出金額統計、degree 特徵、交易次數、`log1p`特徵與推斷是否為玉山銀行帳戶
- 金額缺少容忍：若無金額欄，將以 `amt=1.0` 做處理
- 可重現性：設定 Python / NumPy / PyTorch（含 CUDA）隨機種子

---

## 函式介面

### `set_seed(seed: int = 42) -> None`
設定 Python / NumPy / PyTorch（含 CUDA）隨機種子，在資料處理與切分前呼叫。

### `load_csvs(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
從 `data_dir` 載入三個 CSV，回傳 `(tx, alert, predict)`：
- `tx`     ：交易紀錄
- `alert`  ：警示帳戶名單
- `predict`：待預測帳戶名單

### `resolve_columns(tx, alert, predict) -> Dict[str, str]`
解析並統一欄位命名，回傳欄位對應字典：
- 交易表：`src`, `dst`, `amt`, `from_type`, `to_type`
- 其他表：`alert_acct`, `predict_acct`  
> 找不到 `amt` 時，會新增 `_AMT_` 欄且固定為 `1.0` 作為替代。

### `build_account_features(tx: pd.DataFrame, col: Dict[str, str]) -> pd.DataFrame`
將交易資料整合為帳戶層級之特徵表。輸出欄包含金額統計、degree特徵、交易次數、`log1p` 特徵轉換與 `is_esun`(是否為玉山銀行帳戶)。缺漏值以 `0.0` 補齊。

---

## 帳戶特徵處理

### A. 題目原始特徵

- 帳戶 ID：`from_acct` / `to_acct`
- 金額：`txn_amt`（缺失時以 `1.0` 補齊）  
- 帳戶型別：`from_type`、`to_type`（1 代表 E.SUN帳戶）  
- 其他欄位：日期/時間、貨幣型別、交易通道等

### B. 整合後的帳戶特徵（輸出欄位與定義）
**1) 帳戶ID轉換**
- 由`from_acct` / `to_acct`整理出不同帳戶ID，並將ID轉換成Index來表示，以此建立帳戶特徵向量。

**2) 交易匯出**
- `total_send_amt`：總交易匯出金額  
- `max_send_amt` / `min_send_amt`：單筆匯出最大/最小金額  
- `avg_send_amt`：平均匯出金額  
- `out_deg`：交易匯出涉及的**不同收款帳戶數**
- `out_tx_count`：交易匯出總次數

**3) 交易匯入**
- `total_recv_amt`：總交易匯入金額  
- `max_recv_amt` / `min_recv_amt`：單筆匯入最大/最小金額  
- `avg_recv_amt`：平均匯入金額 
- `in_deg`：交易匯入涉及的**不同匯款帳戶數** 
- `in_tx_count`：交易匯入總次數

**4) `log1p` 特徵轉換（抑制極值）**
- 對下列欄位建立 `log1p_<col>`：  
`total_send_amt`, `total_recv_amt`,  
`max_send_amt`, `min_send_amt`, `avg_send_amt`,  
`max_recv_amt`, `min_recv_amt`, `avg_recv_amt`,  
`out_tx_count`, `in_tx_count`  
> 公式定義：`log1p(x) = ln(1 + x)`

**5) `is_esun`（是否為玉山帳戶）**  
- 由 `from_type`/`to_type`整理出
  - 型別為 **1** 的帳戶，都標記為 `is_esun=1`  
  - 其餘帳戶標記為 `0`  
- 若 `from_type`/`to_type` 不可用，則 **預設為 `1`**

**6) 資料缺失處理**
- 若 `amt` 缺失，設為 `1.0` 仍能統計交易特徵
- 其餘缺失的欄位一律以 `0.0` 填補  

---