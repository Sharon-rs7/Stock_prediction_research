### Table 1: Dataset and Universe Construction Funnel

| Stage / Filter Gate | Criterion / Definition | Survived Tickers | Elimination Rate |
| :--- | :--- | :--- | :--- |
| **0. Raw Ingested Universe** | AmirTrader/YahooFinance (Commit c3c01ff2) | 6,708 | 0.00% |
| **1. Instrument Type Heuristic** | Ordinary Common Stock Candidates (Excl. PFD/Warrant/Unit) | 6,051 | 9.79% |
| **2. Data Quality & Cleansing** | Negative Price = 0, Invalid OHLC = 0, Min Close > $0 | 6,039 | - |
| **3. Non-Stagnant Trading Activity** | Zero-Volume Trading Days $\le$ 1.0% | 5,053 | - |
| **4. Strict Calendar Synchronization** | Exact Panel Balance: 1,759 trading days (2019-09-26 to 2026-09-25) | 3,483 | - |
| **5. Core Market Liquidity** | Median Daily Volume $\ge$ 100,000 shares | **2,435** | - |

*Note: Universe B represents a strictly balanced, survivorship-conditioned panel of 2,435 liquid ordinary common equities.*
