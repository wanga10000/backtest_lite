# 量化交易回測系統專案說明

## 專案概述

這是一個量化交易策略回測系統，主要基於K線數據的策略開發和測試。系統支援多種資產類別（加密貨幣、外匯、貴金屬、股票等）。主要目標是開發出在實盤中能夠穩定獲利的交易策略。

## 核心功能

### 1. 策略回測引擎
- **高效能回測核心**: 使用Cython優化的C++核心，提供快速回測能力
- **多進程並行**: 支援多核心並行處理，適合大規模參數測試
- **支援Tick-based Or K-based模式**
- **完整的手續費計算**: 支援多種手續費模式（點差、固定百分比、固定數量、隔夜費用等）
- **市價單執行**: lite版本僅支援市價單，所有訂單立即執行

### 2. 策略框架
- **模組化設計**: 基於`function_base`類的統一策略接口
- **多時間框架**: 支援多個時間框架同時使用
- **數據整合**: 支援多個數據源整合於同個function使用
- **參數配置**: 通過Excel配置文件調整策略參數、帳戶配置、品種資訊
- **數據擴展**: 不只為K線數據設計，只要數據源有的data就能存取

### 3. 性能分析工具

#### 核心指標
- **風險調整收益**: Sharpe比率、Sortino比率、Omega比率
- **回撤分析**: 最大回撤、平均回撤、回撤持續時間
- **交易統計**: 勝率、盈虧比、交易頻率
- **進階指標**: SQN、邊緣比率、退出效率
- **MEA分析**: 藍月大強力工具，卻也是overfit直達車，使用前請詳閱藍月頻道

#### 分析工具（目前較少使用）
- **Monte Carlo模擬**: 隨機抽樣驗證策略穩定性
- **Walk Forward分析**: 滾動窗口優化
- **PBO分析**: 過度優化檢測
- **指標範圍分析**: 參數敏感性測試

*註：由於認知到過度優化（overfitting）的問題，這些優化工具目前較少使用，重點轉向策略的市場適應性和邏輯合理性。

### 4. 數據管理
- **統一數據格式**: 通過獨立的數據收集專案將不同平台的K線數據統一格式化為Parquet壓縮格式

## 技術架構

### 核心組件
```
├── c_utils/           # Cython優化核心
│   ├── c_backtest_core.pyx      # 主回測引擎
│   ├── c_backtest_mea_core.pyx  # MEA分析引擎
│   └── c_backtest_utils.pyx     # 工具函數
├── function/          # 策略庫
│   ├── function_base.py         # 策略基類
│   └── [strategy_name].py       # 具體策略
├── utils/             # 工具模組
│   ├── backtest.py              # 回測接口
│   ├── perfmetric.py            # 性能指標
│   ├── utils_general.py         # 通用工具
└── └── ...                      # 其它Overfit專用工具

```

### 配置系統
- **Excel配置**: 使用Excel文件管理策略參數
- **多工作表**: general、strategy、backtest、opt_params等
- **動態載入**: 支援運行時參數更新

### 優化技術
- **Cython加速**: 關鍵計算使用C++實現 (比MT5還快)
- **向量化計算**: 大量使用NumPy向量化操作
- **記憶體優化**: 分塊處理大數據集
- **並行處理**: 多進程並行回測 (Opt時使用)

## 使用流程

### 1. 策略開發
```python
# 1. 創建策略類繼承function_base
# 2. 實現必要方法：load_strategy_cfg, indicator_calc, on_tick
# 3. 配置Excel參數文件
# 4. 運行回測測試
```

### 2. 參數優化（目前較少使用）
```python
# 1. 設定參數範圍
# 2. 運行網格搜索或遺傳算法
# 3. Walk Forward驗證
# 4. Monte Carlo穩定性測試
```

### 3. 實盤交易
- **策略部署**: 通過實盤專案將驗證過的策略部署到實際交易，function_base類可以直接共用

## 專案特色

1. **K線專用**: 專為K線數據設計，簡化策略開發流程
2. **模組化**: 清晰的架構設計，易於擴展
3. **實用導向**: 專注於實盤可用的策略，避免過度優化
4. **靈活配置**: 支援多種手續費模式和交易參數
5. **彈性**: 可混合多個timeframe框架和品種去回測

## 開發狀態

專案包含多個開發階段：
- **develop/**: 正在開發的策略（尚未完成或待實盤觀察）
- **fail/**: 測試失敗的策略（out-of-sample表現不佳或實盤效果不好）
- **might_work/**: 有潛力的策略
- **test/**: 各種測試和驗證

## 環境設置

### 編譯Cython核心
在開始使用系統前，需要先編譯Cython核心模組：

```powershell
# 在專案根目錄執行
.\compile_cython.ps1
```

**注意事項**：
- 需要安裝Visual Studio Build Tools（包含C++編譯器）
- 確保Python環境已安裝Cython套件
- 編譯成功後會生成 `.pyd` 檔案在 `c_utils/` 目錄
- 如果修改了 `.pyx` 檔案，需要重新編譯

## 基本使用方法

### 1. 策略開發流程

#### 1.1 創建策略類
策略繼承 `function_base` 類並實現必要的方法：

```python
from function.function_base import function_base
from ta.momentum import RSIIndicator


class my_strategy(function_base):
    def __init__(self, agent_name, log_print):
       self.agent_name = agent_name  # 記錄下來主要給log print用
       self.log_print = log_print    # args: (str, to_tg=1). 此為實盤時可用來print東西的function
        # 初始化策略狀態變數
        pass
    
    def load_general_cfg(self, general_cfg):
        # 載入通用配置（資金、時間框架等）
        self.BALANCE = float(general_cfg['BALANCE'])
        self.USED_TFS = str(general_cfg['USED_TFS']).split(",")
        self.BUFSIZE = list(map(int, str(general_cfg['BUFSIZE']).split(",")))
        self.PAUSE = int(general_cfg['PAUSE'])
    
    def load_strategy_cfg(self, strategy_cfg):
        # 載入策略特定參數
        self.POS_MODE = int(strategy_cfg['POS_MODE'])
        self.LEVERAGE = float(strategy_cfg['LEVERAGE'])
        # ... 其他策略參數
    
    def load_symbols_info(self, symbols_info):
        # 載入交易品種資訊
        self.symbols = [sym_info['SYMBOL'] for sym_info in symbols_info]
        self.MIN_QUOTE = symbols_info[0]['MIN_QUOTE']
        self.MIN_BASE = symbols_info[0]['MIN_BASE']
        self.BASE_STEP = symbols_info[0]['BASE_STEP']
    
    def load_df_sets(self, df_sets):
        # 載入K線數據
        self.df_sets = df_sets
        self.declare_df()
    
    def declare_df(self):
        # 聲明使用的數據框架
        df_set = self.df_sets[0]  # 第一個品種，品種順序依照SYMBOLS而定
        self.df = df_set[0]       # 第一個時間框架，順序依照USED_TFS而定
    
    def indicator_calc(self):
        # 計算技術指標
        if len(self.df) >= self.BUFSIZE[0]:
            # 使用ta庫計算指標
            
            self.df['RSI'] = RSIIndicator(close=self.df['Close'], window=14).rsi()
    
    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        # 核心交易邏輯
        new_orders = []
        cancel_order_ids = []
        
        # 檢查時間框架是否更新
        if kidx_changed_flags[0][0] == 1:  # 主時間框架更新
            idx = kidx_sets[0][0]
            
            # 交易邏輯
            if not cur_positions:  # 無持倉時檢查開倉
                side = self.check_open_condition(idx)
                if side != 0:
                    base = self.calculate_position_size(side, cur_balance, bid, ask)
                    if base is not None:
                        new_orders.append({'base': base})
            
            else:  # 有持倉時檢查平倉
                if self.check_close_condition(idx):
                    new_orders.append({})  # order_dict為空字典，其預設default是market單，平掉pos_id=0的倉位
        
        return new_orders, cancel_order_ids
```

#### 1.2 創建Excel配置文件
Excel配置文件包含多個工作表：

**general工作表**（通用配置）：
- `AGENT`: 策略實例名稱
- `FUNCTION`: 策略類名稱
- `SYMBOLS`: 交易品種（逗號分隔）
- `BALANCE`: 初始資金
- `USED_TFS`: 使用時間框架（逗號分隔，如"1m,1h"）
- `TF_OFFSET`: 各時間框架偏移量（逗號分隔，數量需與USED_TFS一致）
- `BUFSIZE`: 各時間框架緩衝大小（逗號分隔，數量需與USED_TFS一致）
- `PAUSE`: 軟停止標誌（實盤用，1=不能開倉但可關倉，0=正常交易）

**strategy工作表**（策略參數）：
- `AGENT`: 對應的agent名稱
- `POS_MODE`: 倉位模式（0=固定槓桿，1=動態槓桿，2=風險百分比）
- `LEVERAGE`: 槓桿倍數
- `RISK_PER`: 風險百分比
- 其他策略特定參數
- (除了AGENT外，其它都是使用者自行設計)

**backtest工作表**（回測配置）：
- `START_DATE`: 回測開始日期
- `STOP_DATE`: 回測結束日期
- `REAL_TICK`: 是否使用真實tick數據（目前不支援，設為0）
- `NO_COST`: 是否忽略交易成本（1=忽略手續費，0=包含手續費）
- `BASE_LIMIT`: 是否檢查交易量限制（1=檢查MIN_BASE等規則，0=不檢查）
- `DETAIL_MEA`: 是否使用詳細MEA分析（1=使用複雜引擎含MAE/MFE，0=標準引擎）

#### 1.3 運行回測
```python
from utils.utils_general import load_config_xlsx, load_symbols_info_xlsx
from utils.backtest import backtest
from utils.perfmetric import perfmetric
from utils.plot_k import plot_k

# 載入配置
CONFIG_FILE = 'my_strategy.xlsx'
SYMBOLS_INFO_FILE = 'symbols_info_binance_future.xlsx'

agent_dicts_dict = load_config_xlsx(CONFIG_FILE)
agent_dicts_dict = load_symbols_info_xlsx(agent_dicts_dict, SYMBOLS_INFO_FILE)

# 運行回測
agent_dicts_dict = backtest(agent_dicts_dict, MULTI_PROCESSING=0.8, print_progress=1)

# 計算性能指標
agent_dicts_dict, pr_results = perfmetric(agent_dicts_dict)

# 顯示結果
from utils.utils_general import report_pr
report_pr(pr_results)
```

### 2. 策略與回測引擎交互機制

#### 2.1 初始化流程
1. **載入配置**: `load_config_xlsx()` 讀取Excel配置
2. **載入品種資訊**: `load_symbols_info_xlsx()` 讀取交易品種詳細資訊
3. **實例化策略**: 根據`FUNCTION`欄位動態載入策略類
4. **配置載入**: 依序調用 `load_general_cfg()`, `load_strategy_cfg()`, `load_symbols_info()`, `load_df_sets()`
5. **數據準備**: 調用 `declare_df()` 和 `indicator_calc()`

#### 2.2 回測執行流程
回測引擎會逐tick調用 `on_tick()` 方法：

```python
def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
    # 參數說明：
    # cur_sidx: 當前品種索引
    # cur_date_int: 當前時間（整數格式）
    # bid, ask: 當前品種的買賣價
    # kidx_sets: 各品種各時間框架的K線索引
    # kidx_changed_flags: 各時間框架是否更新
    # pending_orders: 待執行訂單列表（用於跨品種訂單，通常不需直接操作）
    # cur_positions: 當前持倉 ({pos_id: position_info})
    # cur_balance: 當前餘額
    
    new_orders = []        # 新訂單列表（僅市價單）
    cancel_order_ids = []  # 待取消的訂單ID列表（用於取消pending_orders中的訂單）
    
    # 策略邏輯實現
    # ...
    
    return new_orders, cancel_order_ids
```

#### 2.3 訂單格式

**基本訂單格式**（僅支援市價單）：
```python
{
    'symbol_idx': 0,           # 品種索引（0=第一個品種，1=第二個品種等）     (default: 0)
    'position_id': 0,          # 倉位ID（0=預設，如網格策略需指定不同ID）     (default: 0)
    'base': 1.5,               # 倉位大小（正數=做多，負數=做空，0=平倉）     (default: 0)
    'tag': 'open'              # 自訂標籤，可用於記錄訂單來源或用途           (default: '')
}
```

**注意事項**：
- lite 版本僅支援市價單，所有訂單會立即執行（或等待該品種tick到來時執行）
- 已移除限價單、觸發單和OCO機制

**跨品種訂單處理**：
- 如果訂單的 `symbol_idx` = 當前tick品種：立即執行
- 如果訂單的 `symbol_idx` ≠ 當前tick品種：
  - 若該品種在本輪tick已出現過：使用該品種最後的價格立即執行
  - 若該品種在本輪tick未出現：加入 pending_orders，等待該品種tick到來時執行

**開倉訂單範例**：
```python
# 市價開多倉（使用預設position_id=0）
{
    'base': 0.1,               # 買入0.1個BTC
}

# 市價開空倉（指定position_id）
{
    'position_id': 1,          # 使用倉位ID=1
    'base': -0.05,             # 賣出0.05個BTC
}

# 帶標籤的市價開倉
{
    'position_id': 2,
    'base': 0.2,               # 買入0.2個BTC
    'tag': 'breakout_entry'    # 自訂標籤
}
```

**平倉訂單範例**：
```python
# 平倉指定倉位
{
    'position_id': 1,          # 平倉ID=1的倉位
    'base': 0                   # base=0表示平倉
}
```

**多倉位管理範例**：
```python
# 使用不同的position_id來管理多個倉位
for i in range(5):
    # 每個倉位使用不同的position_id
    order = {
        'position_id': i,      # 倉位ID
        'base': 0.01,          # 買入0.01個BTC
        'tag': f'grid_{i}'     # 網格標籤
    }
    new_orders.append(order)

# 平倉特定倉位
close_order = {
    'position_id': 2,          # 平倉ID=2的倉位
    'base': 0,                 # base=0表示平倉
    'tag': 'grid_close'
}
```

**注意**：lite版本不支援止損止盈掛單，需在策略邏輯中使用市價單實現：
```python
# 在on_tick中檢查止損止盈條件
if cur_positions:
    for pos_id, pos_info in cur_positions.items():
        float_pnl = pos_info['float_pnl']
        
        # 檢查止損
        if float_pnl < -100:
            new_orders.append({'position_id': pos_id, 'base': 0, 'tag': 'stop_loss'})
        
        # 檢查止盈
        elif float_pnl > 200:
            new_orders.append({'position_id': pos_id, 'base': 0, 'tag': 'take_profit'})
```

#### 2.4 Base概念詳解

**Base計算邏輯**：
- `base` 代表要交易的基礎資產數量
- 正數 = 買入（做多），負數 = 賣出（做空），0 = 平倉
- 實際投入資金 = `abs(base) × 當前價格`

**範例**：
```python
# 假設BTC價格=60000，當前餘額=2355 USD
# 想要1倍槓桿做空

base = -2355 / 60000  # = -0.03925
# 這表示賣出0.03925個BTC，實際投入2355 USD

# 在實盤或cfg有開啟BASE_LIMIT=1時，實際計算時需要遵守步長(Step)和最小量(MIN)交易規則：
base = abs(base)  # 先取絕對值
base = (base // BASE_STEP) * BASE_STEP  # 調整到最小步長
if base < MIN_BASE:  # 檢查最小交易量
    base = MIN_BASE
base = base * -1  # 恢復負號（做空）
```

#### 2.5 倉位管理範例 (可由function agent內自行設計)
三種倉位模式示範：

1. **固定金額模式** (`POS_MODE=0`):
   ```python
   quote = self.LEVERAGE * self.BALANCE
   base = quote / open_price
   ```

2. **動態複利模式** (`POS_MODE=1`):
   ```python
   quote = self.LEVERAGE * cur_balance
   base = quote / open_price
   ```

3. **風險百分比模式** (`POS_MODE=2`):
   ```python
   risk_price = open_price - side * atr_value
   quote = ((self.RISK_PER / 100) / abs(1 - (risk_price/open_price))) * cur_balance
   base = quote / open_price
   ```

#### 2.6 多時間框架使用
```python
def declare_df(self):
    df_set = self.df_sets[0]  # 第一個品種
    self.dfh = df_set[1]      # 小時圖
    self.dfm = df_set[0]      # 分鐘圖

def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
    # 檢查小時圖是否更新
    if kidx_changed_flags[0][1] == 1:
        # 小時圖更新，執行主要邏輯
        # Use kidx_sets[0][1] to get current hour index for self.dfh
        pass
    
    # 檢查分鐘圖是否更新
    if kidx_changed_flags[0][0] == 1:
        # 分鐘圖更新，執行細部邏輯
        # Use kidx_sets[0][0] to get current minute index for self.dfm
        pass
```

### 3. 參考範例

建議參考以下現有策略來學習實現方式：
- `function/test_market.py`: 最簡單的策略範例，純市價單買入賣出

每個策略都展示了不同的實現模式和最佳實踐。

---

## Coding Rules to Follow

- Use English comments. The comment format must be: "# " followed by a sentence that starts with a capital letter.
- Avoid excessive try/except blocks unless necessary. Let errors surface for easier debugging.
- Write concise and clear code; prefer simple, readable constructs.
- When get dict value, not use dict.get(key, default), use [''] and not adding default argument.
- When changing new method, don't need to record how the old method implemented, only need to record what the new method should do.

## Version History
- v1.0.0:
  - Initial release of the lite version of the backtesting system
  - Remove limit/trigger order functionality
  - Remove pending orders system completely
  - Remove OCO (One Cancels Other) mechanism
  - Simplify to market orders only (all orders execute immediately)
  - Remove fields: order_id, order_type, price, oco_orders, oco_cancel_ids, invalid, valid, cancel, side
  - Remove functions: check_pending_order_valid(), check_pending_order_touched()
  - Remove MAKER_FEE (all orders use TAKER_FEE now)
  - Improved performance and memory usage
  
## TODO